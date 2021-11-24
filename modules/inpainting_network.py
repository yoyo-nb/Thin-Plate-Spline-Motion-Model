import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork


class InpaintingNetwork(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """
    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, multi_mask = True, **kwargs):
        super(InpaintingNetwork, self).__init__()

        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        in_features = [max_features, max_features, max_features//2]
        out_features = [max_features//2, max_features//4, max_features//8]
        for i in range(num_down_blocks):
            up_blocks.append(UpBlock2d(in_features[i], out_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        resblock = []
        for i in range(num_down_blocks):
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.resblock = nn.ModuleList(resblock)

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation,align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if not self.multi_mask:
            if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out

    def forward(self, source_image, dense_motion):
        out = self.first(source_image) 
        encoder_map = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)

        output_dict = {}
        output_dict['contribution_maps'] = dense_motion['contribution_maps']
        output_dict['deformed_source'] = dense_motion['deformed_source']

        occlusion_map = dense_motion['occlusion_map']
        output_dict['occlusion_map'] = occlusion_map

        deformation = dense_motion['deformation']
        out_ij = self.deform_input(out.detach(), deformation)
        out = self.deform_input(out, deformation)

        out_ij = self.occlude_input(out_ij, occlusion_map[0].detach())
        out = self.occlude_input(out, occlusion_map[0])

        warped_encoder_maps = []
        warped_encoder_maps.append(out_ij)

        for i in range(self.num_down_blocks):
            
            out = self.resblock[2*i](out)
            out = self.resblock[2*i+1](out)
            out = self.up_blocks[i](out)
            
            encode_i = encoder_map[-(i+2)]
            encode_ij = self.deform_input(encode_i.detach(), deformation)
            encode_i = self.deform_input(encode_i, deformation)
            
            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i+1
            encode_ij = self.occlude_input(encode_ij, occlusion_map[occlusion_ind].detach())
            encode_i = self.occlude_input(encode_i, occlusion_map[occlusion_ind])
            warped_encoder_maps.append(encode_ij)

            if(i==self.num_down_blocks-1):
                break

            out = torch.cat([out, encode_i], 1)

        deformed_source = self.deform_input(source_image, deformation)
        output_dict["deformed"] = deformed_source
        output_dict["warped_encoder_maps"] = warped_encoder_maps

        occlusion_last = occlusion_map[-1]
        if not self.multi_mask:
            occlusion_last = F.interpolate(occlusion_last, size=out.shape[2:], mode='bilinear',align_corners=True)

        out = out * (1 - occlusion_last) + encode_i
        out = self.final(out)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction"] = out

        return output_dict

    def get_encode(self, driver_image, occlusion_map):
        out = self.first(driver_image)
        encoder_map = []
        encoder_map.append(self.occlude_input(out.detach(), occlusion_map[-1].detach()))
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out.detach())
            out_mask = self.occlude_input(out.detach(), occlusion_map[2-i].detach())
            encoder_map.append(out_mask.detach())

        return encoder_map

