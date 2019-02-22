import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
from torchsummary import summary

from config import device, hidden_size, teacher_forcing_ratio, SOS_token
from utils import maskNLLLoss


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.U.state_dict()['weight'])

    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        r = F.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = F.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h


class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        C = Variable(torch.zeros(self.hidden_size)).to(device)
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C


class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal_(self.z1.state_dict()['weight'])
        init.xavier_normal_(self.z2.state_dict()['weight'])
        init.xavier_normal_(self.next_mem.state_dict()['weight'])

    def make_interaction(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        questions = questions.expand_as(facts)
        prevM = prevM.expand_as(facts)

        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=2)

        z = z.view(-1, 4 * embedding_size)

        G = F.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G, dim=-1)

        return G

    def forward(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        G = self.make_interaction(facts, questions, prevM)
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem


class QuestionModule(nn.Module):
    def __init__(self, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, questions, word_embedding):
        '''
        questions.size() -> (#batch, #token)
        word_embedding() -> (#batch, #token, #embedding)
        gru() -> (1, #batch, #hidden)
        '''
        questions = word_embedding(questions)
        _, questions = self.gru(questions)
        questions = questions.transpose(0, 1)
        return questions


class InputModule(nn.Module):
    def __init__(self, hidden_size):
        super(InputModule, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=False)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(vgg19.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal_(param)
        self.dropout = nn.Dropout(0.1)

    def forward(self, images):
        '''
        images.size() -> (#batch, #channel, #height, #width)
        facts.size() -> (#batch, 196, #hidden = #embedding)
        '''
        x = self.cnn(images)
        batch_num, channel_num, row_num, column_num = x.size()  # (-1, 512, 14, 14)
        x = x.view(batch_num, channel_num, row_num * column_num)  # (-1, 512, 196)
        x = x.permute(0, 2, 1)  # (-1, 196, 512)
        x = self.dropout(x)

        h0 = Variable(torch.zeros(2, batch_num, self.hidden_size)).to(device)
        facts, hdn = self.gru(x, h0)
        facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]
        return facts


class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.vocab_size = vocab_size
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal_(param)
        self.out = nn.Linear(hidden_size, vocab_size)
        init.xavier_normal_(self.out.state_dict()['weight'])

    def forward(self, input_step, last_hidden, questions, embedding):
        '''
        input_step.size() -> (#batch, 1, #hidden_size)
        last_hidden.size() -> (#batch, 1, #hidden_size)
        concat.size() -> (#batch, 1, #hidden_size * 2)
        '''
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = embedding(input_step)
        # Forward through unidirectional GRU
        # print('embedded.size(): ' + str(embedded.size()))
        # print('questions.size(): ' + str(questions.size()))
        concat = torch.cat((embedded, questions), dim=-1)
        # print('concat.size(): ' + str(concat.size()))
        output, hidden = self.gru(concat, last_hidden)
        output = F.softmax(self.out(output), dim=1)
        # Return output and final hidden state
        return output, hidden


class DMNPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_hop=3, qa=None):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True)
        init.uniform_(self.word_embedding.state_dict()['weight'], a=-(3 ** 0.5), b=3 ** 0.5)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

        self.input_module = InputModule(hidden_size)
        self.question_module = QuestionModule(hidden_size)
        self.memory = EpisodicMemory(hidden_size)
        self.answer_module = AnswerModule(vocab_size, hidden_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, images, questions, targets, mask):
        '''
        questions.size() -> (#batch, 1, #hidden_size)
        '''
        num_batch = questions.size()[0]

        facts = self.input_module(images)
        questions = self.question_module(questions, self.word_embedding)
        M = questions
        for hop in range(self.num_hop):
            M = self.memory(facts, questions, M)

        '''
        M.size() -> (#batch, 1, #hidden_size)
        '''
        M = self.dropout(M)
        # Set initial decoder hidden state to M
        hidden = M.permute(1, 0, 2)

        # Create initial decoder input (start with SOS tokens for each sentence)
        input = Variable(torch.LongTensor([[SOS_token] for _ in range(num_batch)]).to(device))

        # Initialize variables
        loss = 0
        max_target_len = targets.size()[1]
        preds = Variable(torch.zeros([num_batch, max_target_len], dtype=torch.long).to(device))

        # Determine if we are using teacher forcing this iteration
        if self.training:
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                output, hidden = self.answer_module(input, hidden, questions, self.word_embedding)

                # Teacher forcing: next input is current target
                input = targets[:, t].view(-1, 1)

                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(output, targets[:, t], mask[:, t])
                loss += mask_loss

        else:
            for t in range(max_target_len):
                output, hidden = self.answer_module(input, hidden, questions, self.word_embedding)

                # No teacher forcing: next input is decoder's own current output
                _, topi = output.topk(1)
                input = Variable(torch.LongTensor([[topi[i][0]] for i in range(num_batch)]).to(device))
                # print('input.size(): ' + str(input.size()))
                # print('preds[:, t].size(): ' + str(preds[:, t].size()))
                preds[:, t] = input.view(-1)

                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(output, targets[:, t], mask[:, t])
                loss += mask_loss

        return preds, loss

    def interpret_indexed_tensor(self, var):
        if len(var.size()) == 3:
            # var -> n x #sen x #token
            for n, sentences in enumerate(var):
                for i, sentence in enumerate(sentences):
                    s = ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                    print('{}th of batch, {}th sentence, {}'.format(n, i, s))
        elif len(var.size()) == 2:
            # var -> n x #token
            for n, sentence in enumerate(var):
                s = ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                print('{}th of batch, {}'.format(n, s))
        elif len(var.size()) == 1:
            # var -> n (one token per batch)
            for n, token in enumerate(var):
                s = self.qa.IVOCAB[token.data[0]]
                print('{}th of batch, {}'.format(n, s))


if __name__ == '__main__':
    vocab_size = 15270
    # model = DMNPlus(hidden_size, vocab_size, num_hop=3)
    # model = InputModule(hidden_size).to(device)
    model = AnswerModule(vocab_size, hidden_size).to(device)
    model = model.to(device)
    summary(model, input_size=[(hidden_size,), (hidden_size,), (vocab_size,)])
