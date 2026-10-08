import torch
import torch.nn as nn


class PrismaticVLA(nn.Module):
    # Simplified sketch of the OpenVLA forward pass: __init__ (vision_encoder,
    # projector, tokenizer, llm, bin_to_continuous) is omitted.

    def forward(self, image, instruction):
        # 1. Encode the image into patch features, then project them into the
        #    LLM's embedding dimension so they can sit in the same sequence.
        visual_tokens = self.vision_encoder(image)            # [B, N_v, D_v]
        visual_tokens = self.projector(visual_tokens)         # [B, N_v, D_l]

        # 2. Encode the instruction the ordinary way.
        text_tokens = self.tokenizer(instruction)
        text_embeds = self.llm.embed_tokens(text_tokens)      # [B, N_l, D_l]

        # 3. Fusion is just concatenation along the sequence axis. No
        #    cross-attention module is added; ordinary self-attention inside
        #    the LLM lets every text token attend to every image patch.
        input_embeds = torch.cat([visual_tokens, text_embeds], dim=1)  # [B, N_v + N_l, D_l]

        # 4. inputs_embeds bypasses the tokenizer/embedding table, which is
        #    what allows the visual vectors to enter at all. Action tokens
        #    come out of the ORDINARY LM head -- there is no separate action
        #    head, because 256 rarely-used vocabulary slots were repurposed
        #    as action bins.
        logits = self.llm(inputs_embeds=input_embeds).logits  # [B, N, vocab]

        # 5. Read off the 7 action positions and map bin indices back to
        #    continuous values. (Real inference generates these
        #    autoregressively via predict_action; this is simplified.)
        action_tokens = logits[:, -7:, :].argmax(dim=-1)      # [B, 7]
        return self.bin_to_continuous(action_tokens)          # [B, 7]
