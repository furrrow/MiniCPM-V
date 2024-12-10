import os
import re
import math
import json
import traceback
import argparse
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from decord import VideoReader, cpu, bridge
from PIL import Image
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    # print("num frames:", len(frames))
    return frames

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class PerceptionTestMCQADataset(Dataset):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_name = line['metadata']['video_id']
        mc_questions = line['mc_question']

        # for fmt in self.video_formats:  # Added this line
        fmt = '.mp4'  # custom, we only need mp4
        temp_path = os.path.join(args.video_folder, f"{video_name}{fmt}")
        if os.path.exists(temp_path):
            video_path = temp_path
            video_tensor = load_video(video_path)

        else:
            print("warning, video not found", temp_path)
            return None

        instructs = []
        qids = []
        ops = []
        ans = []
        for q in mc_questions:
            question = q['question']
            qid = q['id']
            options = q['options']
            instruct = f'Question: {question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'

            instructs.append(instruct)
            qids.append(qid)
            ops.append(options)
            ans.append(q['answer_id'])

        return {
            'video': video_tensor[0],
            'video_id': video_name,
            'instructs': instructs,
            'question_ids': qids,
            'options': ops,
            'answer': ans,
        }

def collate_fn(batch):
    vid = [x['video'] for x in batch]
    v_id = [x['video_id'] for x in batch]
    ins = [x['instructs'] for x in batch]
    q_ids = [x['question_ids'] for x in batch]
    ops = [x['options'] for x in batch]
    ans = [x['answer'] for x in batch]
    vid = torch.stack(vid, dim=0)
    return vid, v_id, ins, q_ids, ops, ans


def run_inference(args):
    # parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    # parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--answer-file',
    args.video_folder = "/home/jim/Documents/Projects/perception_test/baselines/data/videos"
    args.question_file = "/home/jim/Documents/Projects/perception_test/baselines/data/mc_question_train.json"
    args.answer_file = "./answer_file.json"

    global MAX_NUM_FRAMES
    MAX_NUM_FRAMES = args.num_frames
    MODEL_PATH = "openbmb/MiniCPM-V-2_6"
    # MODEL_PATH = MODEL_PATH.split("/")[-1].replace("-", "_").lower()
    access_token = "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"

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        token=access_token,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    ).eval().cuda()    # sdpa or flash_attention_2, no eager
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    questions = json.load(open(args.question_file, "r"))
    questions = list(questions.values())
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    dataset = PerceptionTestMCQADataset(questions)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_fn)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    MAX_ITER_SIZE = 250

    # Iterate over each sample in the ground truth file
    for i, (video_tensor, video_id, instructs, question_ids, options, true_ans) in enumerate(tqdm(dataloader)):

        # reduce batch dimension
        frames = video_tensor[0]
        video_id = video_id[0]
        instructs = instructs[0]
        question_ids = question_ids[0]
        options = options[0]
        true_ans = true_ans[0]

        qas = []
        for idx, question in enumerate(instructs):
            letters = ['(A)', '(B)', '(C)']
            question_id = question_ids[idx]
            _options = options[idx]
            q_answer = true_ans[idx]

            # make it such that blank is fed instead of original video tensor:
            blank_video_tensor = torch.zeros_like(video_tensor)
            video_tensor = blank_video_tensor

            # Set decode params for video
            params = {}
            params["use_image_id"] = False
            params["max_slice_nums"] = 2  # use 1 if cuda OOM and video resolution > 448*448

            # First round chat
            warmup_question = "describe what is happening in detail"
            msgs = [{'role': 'user', 'content':  [frames, warmup_question]}]

            answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer
            )
            # print("warmup answer:")
            # print(answer)

            # Second round chat
            # pass history context of multi-turn conversation
            msgs.append({"role": "assistant", "content": [answer]})
            msgs.append({"role": "user", "content": [question]})

            answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer
            )
            # print("question:")
            # print(question)
            # print("actual answer:")
            # print(answer)

            output = answer.replace('answer', '')
            output = output.replace('Answer', '')
            pred_answer = re.findall('\(*[A-C]\)*', output)
            try:
                assert len(
                    pred_answer) >= 1, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(
                    video_id, question, output)
                pred_answer = pred_answer[0].strip()
                # if not pred_answer.startswith('('):
                pred_answer = pred_answer.strip('()')
                pred_answer = f'({pred_answer})'
                pred_idx = letters.index(pred_answer)
            except:
                traceback.print_exc()
                tmp_options = [x.lower() for x in _options]
                if output.lower() in tmp_options:
                    tmp_options = [x.lower() for x in _options]
                    pred_idx = tmp_options.index(output.lower())
                else:
                    pred_idx = 2

            qas.append({'id': question_id, 'question': question, 'answer_id': pred_idx, 'answer': _options[pred_idx],
                        'answer_text': output, 'true_answer': q_answer, "correct": pred_idx == q_answer})

        ans_file.write('\"{}\": {},\n'.format(video_id, json.dumps(qas)))

        if i > min(MAX_ITER_SIZE, len(dataloader)):
            print("max eval size reached, exiting...")
            break
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    # parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    args = parser.parse_args()

    run_inference(args)
