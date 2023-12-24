"""Implementation of CRNN model."""
import timm
from torch import Tensor, nn

RNN_FEATURES_DIM: int = 576


class CRNN(nn.Module):
    """
    A Convolutional Recurrent Neural Network (CRNN) for Optical Character Recognition (OCR).

    This model uses a CNN backbone from the 'timm' library for feature extraction,
    followed by an RNN for sequence modeling, and a fully connected layer for character classification.
    """

    def __init__(
        self,
        encoder: str,
        num_classes: int,
        rnn_hidden_size: int = 48,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.1,
        rnn_features_num: int = 64,
    ):
        """Initialize the CRNN.

        Args:
            encoder (str): cnn backbone to use.
            num_classes (int): number of classes to predict
            rnn_hidden_size (int): hidden size of the model. Defaults to 256.
            rnn_num_layers (int): number of lstm layers to use. Defaults to 2.
            rnn_dropout (float): rnn droput prob. Defaults to 0.1
            rnn_features_num (int): rnn input geatures dim. Defaults to 128
        """
        super().__init__()

        print("num_classes:", num_classes)
        print("rnn_hidden_size:", rnn_hidden_size)
        print("rnn_num_layers:", rnn_num_layers)
        print("rnn_dropout:", rnn_dropout)
        print("rnn_features_num:", rnn_features_num)

        self.backbone = timm.create_model(encoder, pretrained=True, features_only=True, out_indices=(2,))
        backbone_output_dim = self.backbone.feature_info.info[2]["num_chs"]  # noqa: WPS219
        self.gate = nn.Conv2d(backbone_output_dim, rnn_features_num, kernel_size=1, bias=False)

        # Рекуррентная часть.
        self.rnn = nn.GRU(  # type: ignore
            input_size=RNN_FEATURES_DIM,
            hidden_size=rnn_hidden_size,
            dropout=rnn_dropout,
            bidirectional=True,
            batch_first=False,
            num_layers=rnn_num_layers,
        )

        # Fully connected layer for character classification
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)  # *2 for bidirectional
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, image: Tensor) -> Tensor:
        """
        Forward pass of the CRNN model.

        Args:
            image (Tensor): Input tensor of shape [batch, channels, height, width].

        Returns:
            Tensor: Output tensor of shape [batch, timesteps, num_classes].
        """
        # Feature extraction through ResNet18
        features = self.backbone(image)[0]
        features = self.gate(features)

        # Reshape for LSTM
        features = features.permute(3, 0, 2, 1)
        width, batch, height, channels = features.shape
        features = features.reshape(width, batch, height * channels)

        # Sequence modeling through LSTM
        recurrent, _ = self.rnn(features)

        # Character classification
        output = self.fc(recurrent)
        return self.softmax(output)
