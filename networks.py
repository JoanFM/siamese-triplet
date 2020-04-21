import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.out_dimensions = 2
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class MMFashionEmbeddingNet(nn.Module):
    def __init__(self, out_dimensions):
        super(MMFashionEmbeddingNet, self).__init__()
        self.out_dimensions = out_dimensions
        self.convnet = nn.Sequential(nn.Conv2d(3, 16, 3),
                                     nn.PReLU(),
                                     nn.BatchNorm2d(16),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(16, 32, 3),
                                     nn.PReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 3),
                                     nn.PReLU(),
                                     nn.BatchNorm2d(64),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 3, stride=2),
                                     nn.PReLU(),
                                     nn.BatchNorm2d(64),
                                     nn.MaxPool2d(2, stride=2)
                                     )

        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 64 * 4 * 4),
                                nn.PReLU(),
                                nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, out_dimensions)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class MMFashionEmbeddingAlexNet(nn.Module):
    def __init__(self, out_dimensions):
        super(MMFashionEmbeddingNet, self).__init__()
        self.out_dimensions = out_dimensions
        self.alexnet = alexnet(pretrained=True)

        self.alexnet.classifier = nn.Sequential(nn.Linear(64 * 7 * 7, 64 * 4 * 4),
                                nn.PReLU(),
                                nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, out_dimensions)
                                )

        for params in self.alexnet.features.parameters():
            params.requires_grad = False

        for params in list(self.alexnet.features.parameters())[-4:]:
            params.requires_grad = True

    def forward(self, x):
        output = self.alexnet.features(x)
        output = output.view(output.size()[0], -1)
        output = self.alexnet.classifier(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(embedding_net.out_dimensions, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
