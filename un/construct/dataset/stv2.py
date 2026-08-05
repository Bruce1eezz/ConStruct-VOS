from .transforms import *
import os
from glob import glob
from PIL import Image
import torchvision as tv

#这个是直接复制num_frames份
class TestSTV2(torch.utils.data.Dataset):
    def __init__(self, root, num_frames=3):
        self.root = root
        self.num_frames = num_frames

        if self.num_frames % 2 == 0:
            print(
                f"警告: num_frames ({self.num_frames}) 是偶数，中心帧可能无法严格位于正中间。建议使用奇数。已将其调整为 {self.num_frames + 1}。")
            self.num_frames += 1

        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

        self.all_video_info = []

        self.init_data()

        if not self.all_video_info:
            raise RuntimeError("数据集中没有可用的视频信息。请检查路径、文件完整性。")

    def init_data(self):
        img_root_dir = os.path.join(self.root, 'Images')
        if not os.path.exists(img_root_dir):
            raise FileNotFoundError(f"Images 目录不存在: {img_root_dir}")

        self.video_list = sorted(os.listdir(img_root_dir))

        for video_name in self.video_list:
            img_dir = os.path.join(self.root, 'Images', video_name)
            flow_dir = os.path.join(self.root, 'Flows', video_name)
            mask_dir = os.path.join(self.root, 'Annotations', video_name)

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

        all_center_imgs = []
        all_center_flows = []
        all_center_masks = []
        all_ref_imgs_sequences = []
        all_ref_flows_sequences = []
        all_ref_masks_sequences = []

        # 遍历视频中的每一帧，将其视为中心帧
        for center_frame_idx_in_video in range(total_frames):
            try:
                # 只加载中心帧
                img = Image.open(img_paths[center_frame_idx_in_video]).convert('RGB')
                flow = Image.open(flow_paths[center_frame_idx_in_video]).convert('RGB')
                mask = Image.open(mask_paths[center_frame_idx_in_video]).convert('L')
            except IndexError:
                print(f"错误: 视频 {video_name} 帧索引 {center_frame_idx_in_video} 超出范围。检查文件列表是否正确。")
                continue

            # 转换为张量
            current_center_img = self.to_tensor(img)
            current_center_flow = self.to_tensor(flow)
            current_center_mask = self.to_mask(mask)
            current_center_mask = (current_center_mask > 0.5).long()

            # 将中心帧复制 num_frames 次作为参考序列
            stacked_ref_imgs_for_this_center = current_center_img.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
            stacked_ref_flows_for_this_center = current_center_flow.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
            stacked_ref_masks_for_this_center = current_center_mask.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
            stacked_ref_masks_for_this_center = (stacked_ref_masks_for_this_center > 0.5).long()

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
                'center_imgs': torch.empty(0, 3, 512, 512),
                'center_flows': torch.empty(0, 3, 512, 512),
                'center_masks': torch.empty(0, 512, 512, dtype=torch.long),
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