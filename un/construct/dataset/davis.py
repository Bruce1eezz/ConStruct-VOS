from .transforms import *
import os
import random
from glob import glob
from PIL import Image
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainDAVIS(torch.utils.data.Dataset):
    def __init__(self, root, year, split, num_frames=3, max_jump=3, clip_n=128):
        self.root = root
        with open(os.path.join(root, 'ImageSets', '{}/{}.txt'.format(year, split)), 'r') as f:
            self.video_list = f.read().splitlines()
        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()
        self.num_frames = num_frames
        self.max_jump = max_jump
        self.clip_n = clip_n

        self.all_frame_info = []
        self.video_frame_counts = {}

        for video_name in self.video_list:
            img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)

            img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
            current_video_total_frames = len(img_list)
            self.video_frame_counts[video_name] = current_video_total_frames

            if current_video_total_frames < self.num_frames:
                print(
                    f"警告: 视频 {video_name} 总帧数 ({current_video_total_frames}) 少于所需的 {self.num_frames} 帧，将跳过。")
                continue

            for i in range(current_video_total_frames):
                self.all_frame_info.append({
                    'video_name': video_name,
                    'frame_idx_in_video': i,  # 记录帧在视频中的原始索引 (可能被选为中心帧)
                })

        if not self.all_frame_info:
            raise RuntimeError(
                "数据集中没有可用的帧信息。请检查路径、文件完整性以及 num_frames 参数是否过大导致无有效采样点。")

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        while True:
            # 随机选择一个中心帧的信息
            center_frame_info = random.choice(self.all_frame_info)
            video_name = center_frame_info['video_name']
            center_frame_idx_in_video = center_frame_info['frame_idx_in_video']

            current_video_total_frames = self.video_frame_counts[video_name]

            num_left_to_pick = (self.num_frames - 1) // 2
            num_right_to_pick = (self.num_frames - 1) - num_left_to_pick

            selected_frame_indices_in_video = [center_frame_idx_in_video]

            # 尝试向左采样
            current_left_idx = center_frame_idx_in_video
            for _ in range(num_left_to_pick):
                prev_frame_candidate_start = max(0, current_left_idx - (self.max_jump + 1))
                prev_frame_candidate_end = current_left_idx - 1

                if prev_frame_candidate_start > prev_frame_candidate_end:
                    break

                new_left_frame = random.randint(prev_frame_candidate_start, prev_frame_candidate_end)
                selected_frame_indices_in_video.insert(0, new_left_frame)
                current_left_idx = new_left_frame

            # 尝试向右采样
            # 这里的 current_right_idx 应该基于当前已选的最右侧帧，而不是中心帧
            current_right_idx = selected_frame_indices_in_video[-1]

            for _ in range(num_right_to_pick):
                next_frame_candidate_start = current_right_idx + 1
                next_frame_candidate_end = min(current_right_idx + self.max_jump + 1, current_video_total_frames)

                if next_frame_candidate_start >= next_frame_candidate_end:
                    break

                new_right_frame = random.randint(next_frame_candidate_start, next_frame_candidate_end - 1)
                selected_frame_indices_in_video.append(new_right_frame)
                current_right_idx = new_right_frame

            # 补偿逻辑
            if len(selected_frame_indices_in_video) < self.num_frames:
                remaining_frames_to_pick = self.num_frames - len(selected_frame_indices_in_video)

                # 尝试向右补偿
                current_right_idx_for_compensation = selected_frame_indices_in_video[-1]
                for _ in range(remaining_frames_to_pick):
                    next_frame_candidate_start = current_right_idx_for_compensation + 1
                    next_frame_candidate_end = min(current_right_idx_for_compensation + self.max_jump + 1,
                                                   current_video_total_frames)

                    if next_frame_candidate_start >= next_frame_candidate_end:
                        break

                    new_right_frame = random.randint(next_frame_candidate_start, next_frame_candidate_end - 1)
                    selected_frame_indices_in_video.append(new_right_frame)
                    current_right_idx_for_compensation = new_right_frame

                remaining_frames_to_pick = self.num_frames - len(selected_frame_indices_in_video)

                # 尝试向左补偿
                if remaining_frames_to_pick > 0:
                    current_left_idx_for_compensation = selected_frame_indices_in_video[0]
                    for _ in range(remaining_frames_to_pick):
                        prev_frame_candidate_start = max(0, current_left_idx_for_compensation - (self.max_jump + 1))
                        prev_frame_candidate_end = current_left_idx_for_compensation - 1

                        if prev_frame_candidate_start > prev_frame_candidate_end:
                            break

                        new_left_frame = random.randint(prev_frame_candidate_start, prev_frame_candidate_end)
                        selected_frame_indices_in_video.insert(0, new_left_frame)
                        current_left_idx_for_compensation = new_left_frame

                if len(selected_frame_indices_in_video) < self.num_frames:
                    continue  # 重新选择中心帧

            # 对最终选定的帧索引进行排序，以确保它们是递增的
            selected_frame_indices_in_video.sort()

            # 找到中心帧在排序后的 selected_frame_indices_in_video 列表中的新索引
            try:
                # 这个是中心帧原始的索引，我们需要找到它在 selected_frame_indices_in_video 中的位置
                center_frame_position_in_selected = selected_frame_indices_in_video.index(center_frame_idx_in_video)
            except ValueError:
                # 理论上，在预过滤和补偿后，中心帧应该总是在列表中
                # 但以防万一，如果找不到，则重新选择
                print(
                    f"警告: 无法找到中心帧 {center_frame_idx_in_video} 在选定帧列表 {selected_frame_indices_in_video} 中，重新采样。")
                continue  # 重新选择中心帧

            # 找到了一个符合条件的帧序列，现在加载它们
            imgs, flows, masks = [], [], []
            img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
            flow_dir = os.path.join(self.root, 'JPEGFlows', '480p', video_name)
            mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)

            img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
            flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
            mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

            # 打印当前选中的图片位置和名字（所有帧）
            #print(f"视频: {video_name}")
            for i, frame_id in enumerate(selected_frame_indices_in_video):

                """
                print(
                    f"  帧 {i + 1}: 图像文件: {img_list[frame_id]}, "
                    f"光流文件: {flow_list[frame_id]}, 掩码文件: {mask_list[frame_id]}")
                """


                img = Image.open(img_list[frame_id]).convert('RGB')
                flow = Image.open(flow_list[frame_id]).convert('RGB')
                mask = Image.open(mask_list[frame_id]).convert('P')

                # resize to 512p
                img = img.resize((512, 512), Image.BICUBIC)
                flow = flow.resize((512, 512), Image.BICUBIC)
                mask = mask.resize((512, 512), Image.NEAREST)

                imgs.append(img)
                flows.append(flow)
                masks.append(mask)

            # 如果成功获取到所有帧，则跳出循环
            break

        # joint flip (应用于所有帧)
        if random.random() > 0.5:
            imgs = [TF.hflip(im) for im in imgs]
            flows = [TF.hflip(fl) for fl in flows]
            masks = [TF.hflip(mk) for mk in masks]
        if random.random() > 0.5:
            imgs = [TF.vflip(im) for im in imgs]
            flows = [TF.vflip(fl) for fl in flows]
            masks = [TF.vflip(mk) for mk in masks]

        # 提取中心帧数据
        center_img = self.to_tensor(imgs[center_frame_position_in_selected]).unsqueeze(0).unsqueeze(0)
        center_flow = self.to_tensor(flows[center_frame_position_in_selected]).unsqueeze(0).unsqueeze(0)
        center_mask = self.to_mask(masks[center_frame_position_in_selected]).unsqueeze(0).unsqueeze(0)
        center_mask = (center_mask != 0).long()

        # 将所有帧堆叠起来，形成一个 (num_frames, C, H, W) 的张量
        stacked_ref_imgs = torch.stack([self.to_tensor(img) for img in imgs]).unsqueeze(0)
        stacked_ref_flows = torch.stack([self.to_tensor(flow) for flow in flows]).unsqueeze(0)
        # 这里不需要 masks 的 ref_masks，因为您只要求返回 ref_imgs 和 ref_flows
        # 但如果需要，可以添加 stacked_ref_masks = torch.stack([self.to_mask(mask) for mask in masks])
        # stacked_ref_masks = (stacked_ref_masks != 0).long()

        return {
            'center_imgs': center_img,
            'center_flows': center_flow,
            'center_masks': center_mask,
            'ref_imgs': stacked_ref_imgs,
            'ref_flows': stacked_ref_flows
        }

class TestDAVIS(torch.utils.data.Dataset):
    def __init__(self, root, year, split, num_frames=5):
        self.root = root
        self.year = year
        self.split = split
        self.num_frames = num_frames

        if self.num_frames % 2 == 0:
            print(
                f"警告: num_frames ({self.num_frames}) 是偶数，中心帧可能无法严格位于正中间。建议使用奇数。已将其调整为 {self.num_frames + 1}。"
            )
            self.num_frames += 1

        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

        self.all_video_info = []

        self.init_data()

        if not self.all_video_info:
            raise RuntimeError(
                "数据集中没有可用的视频信息。请检查路径、文件完整性。"
            )

    def init_data(self):
        image_sets_path = os.path.join(self.root, 'ImageSets', self.year)
        if not os.path.exists(image_sets_path):
            raise FileNotFoundError(f"ImageSets 目录不存在: {image_sets_path}")

        list_file_path = os.path.join(image_sets_path, f"{self.split}.txt")
        if not os.path.exists(list_file_path):
            raise FileNotFoundError(f"列表文件不存在: {list_file_path}")

        with open(list_file_path, 'r') as f:
            self.video_list = f.read().splitlines()

        for video_name in self.video_list:
            img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
            flow_dir = os.path.join(self.root, 'JPEGFlows', '480p', video_name)
            mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)

            if not os.path.exists(img_dir):
                print(f"警告: 视频 {video_name} 的图像目录不存在，将跳过: {img_dir}")
                continue
            if not os.path.exists(flow_dir):
                print(f"警告: 视频 {video_name} 的光流目录不存在，将跳过: {flow_dir}")
                continue
            if not os.path.exists(mask_dir):
                print(f"警告: 视频 {video_name} 的掩码目录不存在，将跳过: {mask_dir}")
                continue

            img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))
            flow_paths = sorted(glob(os.path.join(flow_dir, '*.jpg')))
            mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))

            current_video_total_frames = len(img_paths)

            if current_video_total_frames == 0:
                print(f"警告: 视频 {video_name} 没有图像帧，将跳过。")
                continue

            self.all_video_info.append({
                'video_name': video_name,
                'total_frames': current_video_total_frames,
                'img_paths': img_paths,
                'flow_paths': flow_paths,
                'mask_paths': mask_paths,
            })

    def __len__(self):
        return len(self.all_video_info)

    def __getitem__(self, idx):
        video_info = self.all_video_info[idx]
        video_name = video_info['video_name']
        total_frames = video_info['total_frames']
        img_paths = video_info['img_paths']
        flow_paths = video_info['flow_paths']
        mask_paths = video_info['mask_paths']

        center_position_in_sequence = self.num_frames // 2

        all_center_imgs = []
        all_center_flows = []
        all_center_masks = []
        all_ref_imgs_sequences = []
        all_ref_flows_sequences = []
        all_ref_masks_sequences = []

        # 遍历视频中的每一帧，将其视为中心帧来采样其对应的参考序列
        for center_frame_idx_in_video in range(total_frames):
            selected_frame_indices = []
            for i in range(self.num_frames):
                frame_offset = i - center_position_in_sequence
                current_frame_idx = center_frame_idx_in_video + frame_offset

                if current_frame_idx < 0:
                    selected_frame_indices.append(0)
                elif current_frame_idx >= total_frames:
                    selected_frame_indices.append(total_frames - 1)
                else:
                    selected_frame_indices.append(current_frame_idx)

            imgs_current_sequence, flows_current_sequence, masks_current_sequence = [], [], []

            # 打印当前中心帧及其选取的参考帧信息
            #print(f"\n--- 视频: {video_name}, 当前中心帧索引: {center_frame_idx_in_video} ---")
            for i, frame_id in enumerate(selected_frame_indices):
                # 确保索引 frame_id 在 img_paths 的有效范围内
                img_filename = os.path.basename(img_paths[frame_id]) if frame_id < len(img_paths) else "N/A"
                flow_filename = os.path.basename(flow_paths[frame_id]) if frame_id < len(flow_paths) else "N/A"
                mask_filename = os.path.basename(mask_paths[frame_id]) if frame_id < len(mask_paths) else "N/A"

                # 标记中心帧
                frame_label = "参考帧"
                if i == center_position_in_sequence:
                    frame_label = "中心帧"
                """
                print(
                    f"  {frame_label} {i + 1} (视频原始索引: {frame_id}): "
                    f"图像文件: {img_filename}, 光流文件: {flow_filename}, 掩码文件: {mask_filename}"
                )                
                """


            for frame_id in selected_frame_indices:
                try:
                    img = Image.open(img_paths[frame_id]).convert('RGB')
                    flow = Image.open(flow_paths[frame_id]).convert('RGB')
                    mask = Image.open(mask_paths[frame_id]).convert('P')
                except IndexError:
                    print(f"错误: 视频 {video_name} 帧索引 {frame_id} 超出范围。检查文件列表是否正确。")
                    continue



                imgs_current_sequence.append(img)
                flows_current_sequence.append(flow)
                masks_current_sequence.append(mask)

            if not imgs_current_sequence:
                continue

            current_center_img = self.to_tensor(imgs_current_sequence[center_position_in_sequence])
            current_center_flow = self.to_tensor(flows_current_sequence[center_position_in_sequence])
            current_center_mask = self.to_mask(masks_current_sequence[center_position_in_sequence])
            if self.year == '2016':
                current_center_mask = (current_center_mask != 0).long()

            stacked_ref_imgs_for_this_center = torch.stack([self.to_tensor(img) for img in imgs_current_sequence])
            stacked_ref_flows_for_this_center = torch.stack([self.to_tensor(flow) for flow in flows_current_sequence])
            stacked_ref_masks_for_this_center = torch.stack([self.to_mask(mask) for mask in masks_current_sequence])
            if self.year == '2016':
                stacked_ref_masks_for_this_center = (stacked_ref_masks_for_this_center != 0).long()

            all_center_imgs.append(current_center_img)
            all_center_flows.append(current_center_flow)
            all_center_masks.append(current_center_mask)
            all_ref_imgs_sequences.append(stacked_ref_imgs_for_this_center)
            all_ref_flows_sequences.append(stacked_ref_flows_for_this_center)
            all_ref_masks_sequences.append(stacked_ref_masks_for_this_center)

        if not all_center_imgs:
            print(f"警告: 视频 {video_name} 未能成功加载任何帧序列。")
            return {
                'video_name': video_name,
                'center_imgs': torch.empty(0, 3, 512, 512), # 给出预期形状
                'center_flows': torch.empty(0, 3, 512, 512),
                'center_masks': torch.empty(0, 512, 512, dtype=torch.long), # mask的dtype
                'ref_imgs': torch.empty(0, self.num_frames, 3, 512, 512),
                'ref_flows': torch.empty(0, self.num_frames, 3, 512, 512),
                'ref_masks': torch.empty(0, self.num_frames, 512, 512, dtype=torch.long)
            }


        video_center_imgs = torch.stack(all_center_imgs).unsqueeze(1)
        video_center_flows = torch.stack(all_center_flows).unsqueeze(1)
        video_center_masks = torch.stack(all_center_masks).unsqueeze(1)

        video_ref_imgs = torch.stack(all_ref_imgs_sequences)
        video_ref_flows = torch.stack(all_ref_flows_sequences)
        video_ref_masks = torch.stack(all_ref_masks_sequences)

        return {
            'video_name': video_name,
            'center_imgs': video_center_imgs,
            'center_flows': video_center_flows,
            'center_masks': video_center_masks,
            'ref_imgs': video_ref_imgs,
            'ref_flows': video_ref_flows,
            'ref_masks': video_ref_masks
        }

#这个是为了外挂时序，修改的dataloader

"""
统一成[B,L,S,C,H,W], S为选择的参考帧数量num

TrianDAVIS:

    imgs的尺寸: torch.Size([B, 1, 1, 3, 512, 512])
    flows的尺寸: torch.Size([B, 1, 1, 3, 512, 512])
    masks的尺寸: torch.Size([B, 1, 1, 1, 512, 512])
    ref_imgs的尺寸: torch.Size([B, 1, num, 3, 512, 512])
    ref_flows的尺寸: torch.Size([B, 1, num, 3, 512, 512])

TestDAVIS:

    center_imgs 的尺寸: torch.Size([1, L, 1, 3, 480, 854])
    center_flows 的尺寸: torch.Size([1, L, 1, 3, 480, 856])
    center_masks 的尺寸: torch.Size([1, L, 1, 1, 480, 854])
    ref_imgs 的尺寸: torch.Size([1, L, num, 3, 480, 854])
    ref_flows 的尺寸: torch.Size([1, L, num, 3, 480, 856])
    ref_masks 的尺寸: torch.Size([1, L, num, 1, 480, 854])

"""