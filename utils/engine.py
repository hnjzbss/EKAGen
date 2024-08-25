import torch
import math
import sys
import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor as meteor
from pycocoevalcap.rouge.rouge import Rouge as rouge
from models import utils


def train_one_epoch(model, tmodel, class_model, criterion, criterionKD, data_loader,
                    optimizer, device, max_norm, thresholds, tokenizer, config):
    model.train()
    criterion.train()
    class_model.eval()
    tmodel.eval()

    epoch_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, com_images, com_masks, caps, cap_masks, image_class in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            com_samples = utils.NestedTensor(com_images, com_masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            logit = class_model(image_class.to(device))
            thresholded_predictions = 1 * (logit.cpu().numpy() > thresholds)
            t_outputs = tmodel(samples, caps[:, :-1], cap_masks[:, :-1], [thresholded_predictions, tokenizer])
            outputs = model(com_samples, caps[:, :-1], cap_masks[:, :-1], [thresholded_predictions, tokenizer])
            kd_loss = criterionKD(outputs, t_outputs.detach()) * config.delta

            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:]) + kd_loss
            loss_value = loss.item()
            epoch_loss += loss_value

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)

    return epoch_loss / total


def create_caption_and_mask(start_token, max_length, batch_size):
    caption_template = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask_template = torch.ones((batch_size, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


def compute_scores(gts, res):
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (meteor(), "METEOR"),
        (rouge(), "ROUGE_L")
    ]
    eval_res = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, _ = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


@torch.no_grad()
def evaluate(model, class_model, criterion, data_loader, device, config, thresholds, tokenizer):
    model.eval()
    criterion.eval()
    class_model.eval()
    total = len(data_loader)
    caption_list = []
    caption_tokens_list = []

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, _, _, caps, _, image_class in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caption, cap_mask = create_caption_and_mask(
                config.start_token, config.max_position_embeddings, config.batch_size)
            try:
                for i in range(config.max_position_embeddings - 1):
                    logit = class_model(image_class.to(device))
                    thresholded_predictions = 1 * (logit.cpu().numpy() > thresholds)
                    predictions = model(samples.to(device), caption.to(device), cap_mask.to(device),
                                        [thresholded_predictions, tokenizer])
                    predictions = predictions[:, i, :]
                    predicted_id = torch.argmax(predictions, axis=-1)
                    if i == config.max_position_embeddings - 2:
                        caption_list.extend(caption.cpu().numpy().tolist())
                        caption_tokens_list.extend(caps[:, 1:].cpu().numpy().tolist())
                        break
                    caption[:, i + 1] = predicted_id
                    cap_mask[:, i + 1] = False
            except:
                pass
            pbar.update(1)

        pred = caption_list
        report = caption_tokens_list
        preds_orign = []
        preds = []
        reports = []
        for preds_sentence in pred:
            single_sentence = list()
            for item in preds_sentence:
                single_sentence.append(item)
                if item == 2:
                    preds_orign.append(single_sentence)
                    continue
        for preds_sentence in pred:
            preds.append([item for item in preds_sentence if item not in [config.start_token, config.end_token, 0]])
        for reports_sentence in report:
            reports.append([item for item in reports_sentence if item not in [config.start_token, config.end_token, 0]])
        ground_truth = [tokenizer.decode(item) for item in reports]
        pred_result = [tokenizer.decode(item) for item in preds]
        val_met = compute_scores({i: [gt] for i, gt in enumerate(ground_truth)},
                                 {i: [re] for i, re in enumerate(pred_result)})
        return val_met
