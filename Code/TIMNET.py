import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        # 패딩을 적용하여 causal (non-anticipating) 효과를 만듭니다.
        
        pad_size = (self.conv1.kernel_size[0] - 1) * self.dilation
        
        x = F.pad(x, (pad_size, 0))
        return self.conv1(x)

class SpatialDropout1D(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout2d = nn.Dropout2d(p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = self.dropout2d(x)  # dropout along the feature dimension
        x = x.squeeze(2)  # (N, T, K)
        return x

class SamePadConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(SamePadConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2  # 'same' padding
        self.conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

    def forward(self, x):
        return self.conv(x)

class Temporal_Aware_Block(nn.Module):
    def __init__(self, C, kernel_size, i, dropout_rate=0.1):
        super(Temporal_Aware_Block, self).__init__()

        self.block = nn.Sequential(
            CausalConv1d(in_channels=C, out_channels=C, kernel_size=kernel_size,
                      dilation=i),
            nn.BatchNorm1d(C),
            nn.ReLU(),
            SpatialDropout1D(dropout_rate),

            CausalConv1d(in_channels=C, out_channels=C, kernel_size=kernel_size,
                      dilation=i),
            nn.BatchNorm1d(C),
            nn.ReLU(),
            SpatialDropout1D(dropout_rate)
        )
        self.conv_1x1 = SamePadConv1d(in_channels=C, out_channels=C, kernel_size=1, stride = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_x = x

        output = self.block(x)
        
        if original_x.shape[-1] != output.shape[-1]:
            original_x = self.conv_1x1(original_x)

        output = self.sigmoid(output)
        
        F_x = original_x * output
        return F_x

class WeightLayer(nn.Module):
    def __init__(self, input_dim):
        super(WeightLayer, self).__init__()
        self.kernel = nn.Parameter(torch.randn(input_dim, 1))

    def forward(self, x):
        tempx = x
        
        x = torch.matmul(tempx, self.kernel)
        x = torch.squeeze(x, -1)
        
        return x

class TIMNET(nn.Module):
    def __init__(self, datashape, nb_filters=39, kernel_size=2, nb_stacks=1, dilations=8, num_classes = 8,
                    dropout_rate=0.1, return_sequences=True, name='TIMNET'):
        super(TIMNET, self).__init__()
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.datashape = datashape
        self.num_classes = num_classes

        self.supports_masking = True
        self.mask_value = 0.
        
        self.conv1x1 = CausalConv1d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=1,
                                  dilation=1)

        self.temporal_blocks = nn.ModuleList()
        for i in [2 ** i for i in range(self.dilations)]:
            self.temporal_blocks.append(Temporal_Aware_Block(self.nb_filters, self.kernel_size, i,
                                                                self.dropout_rate))
                
        self.temporal_blocks2 = nn.ModuleList()
        for i in [2 ** i for i in range(self.dilations)]:
            self.temporal_blocks2.append(Temporal_Aware_Block(self.nb_filters, self.kernel_size, i,
                                                                self.dropout_rate))
                
        self.weightlayer = WeightLayer(self.dilations)

        self.classifier = nn.Linear(self.datashape[1], self.num_classes)
        #self.softmax = nn.Softmax(dim = -1)

    def forward(self, inputs):
        inputs = torch.permute(inputs, (0, 2, 1))
        forward = inputs
        backward = torch.flip(inputs, dims=[-1])
        
        forward_convd = self.conv1x1(forward)
        backward_convd = self.conv1x1(backward)

        final_skip_connection = []

        skip_out_forward = forward_convd
        skip_out_backward = backward_convd
        
        for temporal_block, temporal_block2 in zip(self.temporal_blocks, self.temporal_blocks2):
            skip_out_forward = temporal_block(skip_out_forward)
            skip_out_backward = temporal_block2(skip_out_backward)

            temp_skip = skip_out_forward + skip_out_backward
            temp_skip = torch.mean(temp_skip, dim=-1, keepdim=True)
            final_skip_connection.append(temp_skip)

        output_2 = final_skip_connection[0]
        for i, item in enumerate(final_skip_connection):
            if i == 0:
                continue
            output_2 = torch.cat([output_2, item], dim=-1)
        x = output_2
        
        x = self.weightlayer(x)
        
        x = self.classifier(x)
        x = F.log_softmax(x, dim = 1)

        return x
    
