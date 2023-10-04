import copy
import os.path as osp
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

from torch.nn.parallel import DistributedDataParallel as DDP


# from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
# from sampling import cifar_iid, cifar_noniid


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        #n_ctx = cfg.TRAINER.COOP.N_CTX
        n_ctx = cfg.TRAINER.PROMPTFL.N_CTX
        ctx_init = cfg.TRAINER.PROMPTFL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.PROMPTFL.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class PadPrompter(nn.Module):
    def __init__(self, cfg, device, image_size=224):
        super(PadPrompter, self).__init__()
        pad_size = cfg.visual_prompt_size
        self.device = device

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, cfg, device, image_size=224):
        super(FixedPatchPrompter, self).__init__()
        self.isize = image_size
        self.psize = cfg.visual_prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))
        self.device = device

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).to(self.device)
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, cfg, device, image_size=224):
        super(RandomPatchPrompter, self).__init__()
        self.device = device
        self.isize = image_size
        self.psize = cfg.visual_prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).to(self.device)
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt


class MultiModalSelfAttention(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super(MultiModalSelfAttention, self).__init__()
        self.visual_query_weight = nn.Parameter(torch.randn(visual_dim))
        self.visual_key_weight = nn.Parameter(torch.randn(visual_dim))
        self.visual_value_weight = nn.Parameter(torch.randn(visual_dim))
        self.text_query_weight = nn.Parameter(torch.randn(text_dim))
        self.text_key_weight = nn.Parameter(torch.randn(text_dim))
        self.text_value_weight = nn.Parameter(torch.randn(text_dim))

    def forward(self, visual_prompt, text_prompt):
        visual_prompt_flat = visual_prompt.view(1, -1)
        text_prompt_flat = text_prompt.view(1, -1)

        visual_query = visual_prompt_flat * self.visual_query_weight
        text_query = text_prompt_flat * self.text_query_weight
        visual_key = visual_prompt_flat * self.visual_key_weight
        text_key = text_prompt_flat * self.text_key_weight
        visual_value = visual_prompt_flat * self.visual_value_weight
        text_value = text_prompt_flat * self.text_value_weight

        visual_attention_score = torch.matmul(visual_query, visual_key.T)
        text_attention_score = torch.matmul(text_query, text_key.T)

        attention_score = F.softmax(torch.cat([visual_attention_score, text_attention_score]), dim=0)

        meta_prompt = attention_score[0] * visual_value + attention_score[1] * text_value

        return meta_prompt


class CustomCLIP_VandT(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)

        # self.visual_prompt_learner = PadPrompter(cfg, device)
        self.visual_prompt_learner = FixedPatchPrompter(cfg, device)
        # self.visual_prompt_learner = RandomPatchPrompter(cfg, device)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        prompted_image = self.visual_prompt_learner(image)
        image_features = self.image_encoder(prompted_image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return image_features, text_features, logits


@TRAINER_REGISTRY.register()
class PromptFL(TrainerX):
    def check_cfg(self, cfg):
        #assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.PROMPTFL.PREC in ["fp16", "fp32", "amp"]
        pass


    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        #if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
        if cfg.TRAINER.PROMPTFL.PREC == "fp32" or cfg.TRAINER.PROMPTFL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        if cfg.TRAINER.PROMPTFL.VANDT:
            print("Building custom CLIP_VandT")
            self.model = CustomCLIP_VandT(cfg, classnames, clip_model, self.device).to(self.device)
            self.global_model = CustomCLIP_VandT(cfg, classnames, clip_model, self.device).to(self.device)
            self.previous_models = []

            print("Turning off gradients in both the image and the text encoder")
            for name, param in self.model.named_parameters():
                # print(name,":",param.size())
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
            print(f"# params: {count_num_param(self.model):,}")
            print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")

            if cfg.MODEL.INIT_WEIGHTS:
                load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            # NOTE: only give prompt_learner to the optimizer
            self.optim_text = build_optimizer(self.model, cfg.OPTIM, [{'params': self.model.prompt_learner.parameters()}])
            self.optim_visual = build_optimizer(self.model, cfg.OPTIM, [{'params': self.model.visual_prompt_learner.parameters()}])
            self.sched_text = build_lr_scheduler(self.optim_text, cfg)
            self.sched_visual = build_lr_scheduler(self.optim_visual, cfg)
            self.register_model("prompt_learner", self.model.prompt_learner, self.optim_text, self.sched_text)
            self.register_model("visual_prompt_learner", self.model.visual_prompt_learner, self.optim_visual, self.sched_visual)
        else:
            print("Building custom CLIP")
            self.model = CustomCLIP(cfg, classnames, clip_model).to(self.device)

            print("Turning off gradients in both the image and the text encoder")
            for name, param in self.model.named_parameters():
                # print(name,":",param.size())
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
            print(f"# params: {count_num_param(self.model):,}")
            print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")


            if cfg.MODEL.INIT_WEIGHTS:
                load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

            # NOTE: only give prompt_learner to the optimizer

            self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)

            self.sched = build_lr_scheduler(self.optim, cfg)
            self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model, device_ids=[1])


    def forward_backward(self, batch, usr_idx, global_epoch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTFL.PREC
        if self.cfg.TRAINER.PROMPTFL.VANDT:
            if global_epoch == 0:
                if prec == "amp":
                    with autocast():
                        _, _, output = self.model(image)
                        loss = F.cross_entropy(output, label)
                    self.optim_text.zero_grad()
                    self.scaler.scale(loss).backward(retain_graph=True)
                    self.scaler.step(self.optim_text)

                    self.optim_visual.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim_visual)
                    self.scaler.update()
                else:
                    _, _, output = self.model(image)
                    loss = F.cross_entropy(output, label)
                    self.model_backward_and_update(loss)
                loss_summary = {
                    "loss": loss.item(),
                    "acc": compute_accuracy(output, label)[0].item(),
                }
            else:
                if self.cfg.TRAINER.PROMPTFL.ONLY_VISUAL:
                    with autocast(dtype=torch.float16):
                        _, _, output = self.model(image)
                        loss_sup = F.cross_entropy(output, label)
                        self.optim_text.zero_grad()
                        self.scaler.scale(loss_sup).backward(retain_graph=True)
                        self.scaler.step(self.optim_text)

                        self.optim_visual.zero_grad()
                        self.scaler.scale(loss_sup).backward()
                        self.scaler.step(self.optim_visual)
                        self.scaler.update()

                        loss_summary = {
                            "loss": loss_sup.item(),
                            "acc": compute_accuracy(output, label)[0].item(),
                        }
                else:
                    T = 2
                    neta = 0.2
                    cos = nn.CosineSimilarity(dim=-1)
                    if prec == "amp":
                        with autocast(dtype=torch.float16):
                            image_features, text_features, output = self.model(image)
                            loss_sup = F.cross_entropy(output, label)

                            # For Knowledge Distillation
                            # teacher_output = self.global_model(image)
                            # student_output = output
                            # loss_2 = F.kl_div(F.log_softmax(student_output / T, dim=1),
                            #                   F.softmax(teacher_output / T, dim=1),
                            #                   reduction='batchmean') * T ** 2
                            global_image_features, global_text_features, _ = self.global_model(image)

                            pos_vis = cos(image_features, global_image_features)
                            logits_vis = pos_vis.reshape(-1, 1)
                            pos_text = cos(text_features, global_text_features)
                            logits_text = pos_text.reshape(-1, 1)

                            labels_vis = torch.zeros(image.size(0)).to(self.device).long()
                            labels_text = torch.zeros(logits_text.size(0)).to(self.device).long()

                            for previous_model in self.previous_models[usr_idx][-1:]:
                                # previous_model = self.previous_models[usr_idx][-1]
                                previous_model.to(self.device)

                                old_image_features, old_text_features, _ = previous_model(image)

                                neg_vis = cos(image_features, old_image_features)
                                logits_vis = torch.cat((logits_vis, neg_vis.reshape(-1, 1)), dim=1)

                                neg_text = cos(text_features, old_text_features)
                                logits_text = torch.cat((logits_text, neg_text.reshape(-1, 1)), dim=1)

                                torch.cuda.empty_cache()

                            logits_vis = logits_vis / T
                            logits_text = logits_text / T

                            loss_vis = neta * F.cross_entropy(logits_vis, labels_vis) + loss_sup
                            loss_text = neta * F.cross_entropy(logits_text, labels_text) + loss_sup

                        self.optim_text.zero_grad()
                        self.scaler.scale(loss_text).backward(retain_graph=True)
                        self.scaler.step(self.optim_text)

                        self.optim_visual.zero_grad()
                        self.scaler.scale(loss_vis).backward()
                        self.scaler.step(self.optim_visual)
                        self.scaler.update()
                    else:
                        _, _, output = self.model(image)
                        loss = F.cross_entropy(output, label)
                        self.model_backward_and_update(loss)

                    loss_summary = {
                        "loss": loss_text.item(),
                        "acc": compute_accuracy(output, label)[0].item(),
                    }
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
                # self.optim_text.step()
                # self.optim_visual.step()


        else:
            if prec == "amp":
                # print(self.model.state_dict()['prompt_learner.ctx'])
                with autocast():
                    output = self.model(image)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                _, _, output = self.model(image)
                loss = F.cross_entropy(output, label)
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }

            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        if self.cfg.TRAINER.PROMPTFL.VANDT:
            for previous_model in self.previous_models[usr_idx]:
                previous_model.to('cpu')

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model bash main.sh caltech101 rn50_ep50 end 16 1 Falsenot found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


@TRAINER_REGISTRY.register()
class Baseline(TrainerX):
    """Supervised Baseline."""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTFL.PREC == "fp32" or cfg.TRAINER.PROMPTFL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # print(name,":",param.size())
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL.PREC == "amp" else None


    def forward_backward(self, batch, usr_idx, global_epoch):
        input, label = self.parse_batch_train(batch)
        output = self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label




