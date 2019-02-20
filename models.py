import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
from torchsummary import summary

from config import device, hidden_size, batch_size, max_target_len


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
        C = Variable(torch.zeros(self.hidden_size)).cuda()
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

        h0 = Variable(torch.zeros(2, batch_num, self.hidden_size).cuda())
        facts, hdn = self.gru(x, h0)
        facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]
        return facts


class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal_(param)
        self.linear = nn.Linear(hidden_size, vocab_size)
        init.xavier_normal_(self.linear.state_dict()['weight'])

    def forward(self, M, questions, word_embedding):
        '''
        M.size() -> (#batch, 1, #hidden_size)
        questions.size() -> (#batch, 1, #hidden_size)
        '''
        M = self.dropout(M)
        print('M.size(): ' + str(M.size()))
        hidden = M
        batch_size = M.size()[0]

        answer = torch.zeros([batch_size, max_target_len], dtype=torch.long)

        for t in range(max_target_len):
            '''
            preds.size() -> (#batch, 1, #vocab_size)
            topi.size() -> (#batch, 1)
            input.size() -> (#batch, 1, #vocab_size)
            '''
            preds = F.softmax(self.linear(hidden), dim=-1)
            print('preds.size(): ' + str(preds.size()))
            _, topi = preds.topk(1)
            topi = topi.view((batch_size, 1))
            input = word_embedding(topi)
            print('topi.size(): ' + str(topi.size()))
            print('input.size(): ' + str(input.size()))
            print('questions.size(): ' + str(questions.size()))
            concat = torch.cat([input, questions], dim=2).squeeze(1)
            print('concat.size(): ' + str(concat.size()))
            _, hidden = self.gru(concat, hidden)
            print('hidden.size(): ' + str(hidden.size()))
            answer[t] = topi

        return answer


class DMNPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_hop=3, qa=None):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True).cuda()
        init.uniform_(self.word_embedding.state_dict()['weight'], a=-(3 ** 0.5), b=3 ** 0.5)
        self.criterion = nn.CrossEntropyLoss(size_average=False)

        self.input_module = InputModule(hidden_size)
        self.question_module = QuestionModule(hidden_size)
        self.memory = EpisodicMemory(hidden_size)
        self.answer_module = AnswerModule(vocab_size, hidden_size)

    def forward(self, images, questions):
        '''
        contexts.size() -> (#batch, #sentence, #token) -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #token) -> (#batch, 1, #hidden)
        '''
        facts = self.input_module(images)
        questions = self.question_module(questions, self.word_embedding)
        M = questions
        for hop in range(self.num_hop):
            M = self.memory(facts, questions, M)
        preds = self.answer_module(M, questions, self.word_embedding)
        return preds

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

    def get_loss(self, images, questions, targets):
        output = self.forward(images, questions)
        loss = self.criterion(output, targets)
        reg_loss = 0
        for param in self.parameters():
            reg_loss += 0.001 * torch.sum(param * param)
        preds = F.softmax(output, dim=-1)
        _, pred_ids = torch.max(preds, dim=1)
        corrects = (pred_ids.data == targets.data)
        acc = torch.mean(corrects.float())
        return loss + reg_loss, acc


if __name__ == '__main__':
    vocab_size = 15270
    # model = DMNPlus(hidden_size, vocab_size, num_hop=3)
    # model = InputModule(hidden_size).to(device)
    model = AnswerModule(vocab_size, hidden_size).to(device)
    model = model.to(device)
    summary(model, input_size=[(hidden_size,), (hidden_size,), (vocab_size, )])
