import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class LayoutEncoder(nn.Module):
    def __init__(self, layout_encoding_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(LayoutEncoder, self).__init__()
        self.label_encoder = nn.Embedding(vocab_size, layout_encoding_size)
        self.location_encoder = nn.Linear(4, layout_encoding_size)
        self.lstm = nn.LSTM(layout_encoding_size, hidden_size, num_layers, batch_first=True)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.label_encoder.weight.data.uniform_(-0.1, 0.1)
        self.location_encoder.weight.data.uniform_(-0.1, 0.1)
        self.location_encoder.bias.data.fill_(0)

    def forward(self, label_seqs, location_seqs, lengths):
        # sort label sequences and location sequences in batch dimension according to length
        batch_idx = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)
        reverse_batch_idx = torch.LongTensor([batch_idx.index(i) for i in range(len(batch_idx))])

        lens_sorted = sorted(lengths, reverse=True)
        label_seqs_sorted = torch.index_select(label_seqs, 0, torch.LongTensor(batch_idx))
        location_seqs_sorted = torch.index_select(location_seqs, 0, torch.LongTensor(batch_idx))

        # assert torch.equal(torch.index_select(label_seqs_sorted, 0, reverse_batch_idx), label_seqs)
        # assert torch.equal(torch.index_select(location_seqs_sorted, 0, reverse_batch_idx), location_seqs)

        if torch.cuda.is_available():
            reverse_batch_idx = reverse_batch_idx.cuda()
            label_seqs_sorted = label_seqs_sorted.cuda()
            location_seqs_sorted = location_seqs_sorted.cuda()

        # create Variables
        label_seqs_sorted_var = Variable(label_seqs_sorted, requires_grad=False)
        location_seqs_sorted_var = Variable(location_seqs_sorted, requires_grad=False)

        # encode label sequences
        label_encoding = self.label_encoder(label_seqs_sorted_var)

        # encode location sequences
        location_seqs_sorted_var = location_seqs_sorted_var.view(-1, 4)
        location_encoding = self.location_encoder(location_seqs_sorted_var)
        location_encoding = location_encoding.view(label_encoding.size(0), -1, location_encoding.size(1))

        # layout encoding - batch_size x max_seq_len x embed_size
        layout_encoding = label_encoding + location_encoding
        packed = pack(layout_encoding, lens_sorted, batch_first=True)
        hiddens, _ = self.lstm(packed)

        # unpack hiddens and get last hidden vector
        hiddens_unpack = unpack(hiddens, batch_first=True)[0]  # batch_size x max_seq_len x embed_size
        last_hidden_idx = torch.zeros(hiddens_unpack.size(0), 1, hiddens_unpack.size(2)).long()
        for i in range(hiddens_unpack.size(0)):
            last_hidden_idx[i, 0, :] = lens_sorted[i] - 1
        if torch.cuda.is_available():
            last_hidden_idx = last_hidden_idx.cuda()
        last_hidden = torch.gather(hiddens_unpack, 1, Variable(last_hidden_idx, requires_grad=False))  # batch_size x 1 x embed_size
        last_hidden = torch.squeeze(last_hidden, 1)  # batch_size x embed_size

        # convert back to original batch order
        last_hidden = torch.index_select(last_hidden, 0, Variable(reverse_batch_idx, requires_grad=False))

        return last_hidden
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()