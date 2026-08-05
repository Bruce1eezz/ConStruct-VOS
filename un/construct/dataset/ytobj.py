from .transforms import *
import os
from glob import glob
from PIL import Image
import torchvision as tv


class TestYTOBJ(torch.utils.data.Dataset):
    def __init__(self, root, num_frames=5):
        self.root = root
        self.num_frames = num_frames

        if self.num_frames % 2 == 0:
            print(
                f"警告: num_frames ({self.num_frames}) 是偶数，中心帧可能无法严格位于正中间。建议使用奇数。已将其调整为 {self.num_frames + 1}。")
            self.num_frames += 1

        self.video_list = []
        class_list = sorted(os.listdir(os.path.join(root, 'JPEGImages')))
        for class_name in class_list:
            video_list = sorted(os.listdir(os.path.join(root, 'JPEGImages', class_name)))
            for video_name in video_list:
                self.video_list.append(class_name + '_' + video_name)
        self.to_tensor = tv.transforms.ToTensor()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        class_name = self.video_list[idx].split('_')[0]
        video_name = self.video_list[idx].split('_')[1]
        img_dir = os.path.join(self.root, 'JPEGImages', class_name, video_name)
        flow_dir = os.path.join(self.root, 'JPEGFlows', class_name, video_name)
        mask_dir = os.path.join(self.root, 'Annotations', class_name, video_name)
        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        all_center_imgs = []
        all_center_flows = []
        all_center_masks = []
        all_ref_imgs_sequences = []
        all_ref_flows_sequences = []
        all_ref_masks_sequences = []

        # 遍历视频中的每一帧，将其视为中心帧
        for i in range(len(img_list)):
            try:
                # 只加载中心帧
                img = Image.open(img_list[i]).convert('RGB')
                flow = Image.open(flow_list[i]).convert('RGB')
                mask = Image.open(mask_list[i]).convert('L')
            except (IndexError, FileNotFoundError) as e:
                print(f"错误: 视频 {class_name}_{video_name} 帧索引 {i} 无法加载: {e}")
                continue

            # 转换为张量
            current_center_img = self.to_tensor(img)
            current_center_flow = self.to_tensor(flow)
            current_center_mask = self.to_tensor(mask)
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
            print(f"警告: 视频 {class_name}_{video_name} 未能成功加载任何帧序列。")
            return {
                'video_name': f"{class_name}_{video_name}",
                'class_name': class_name,
                'files': mask_list,
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
            'video_name': f"{class_name}_{video_name}",
            'class_name': class_name,
            'files': mask_list,
            'center_imgs': video_center_imgs,
            'center_flows': video_center_flows,
            'center_masks': video_center_masks,
            'ref_imgs': video_ref_imgs,
            'ref_flows': video_ref_flows,
            'ref_masks': video_ref_masks
        }
