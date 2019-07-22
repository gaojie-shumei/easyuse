import torch as tc
import torchvision as tcv
import torch.nn as tcn
import torch.nn.functional as F
import torch.optim as tcoptim
import numpy as np
def next_batch(x_train,y_train,position,batch_size,shuffle=True,randomstate=np.random.randint(0,100)):
    temp_x,temp_y = x_train[position:],y_train[position:]
    if shuffle:
        np.random.seed(randomstate)
        np.random.shuffle(temp_x)
        np.random.seed(randomstate)
        np.random.shuffle(temp_y)
    x_train = np.r_[x_train[0:position],temp_x]
    y_train = np.r_[y_train[0:position],temp_y]
    if batch_size<temp_x.shape[0]:
        batch_x = temp_x[0:batch_size]
        batch_y = temp_y[0:batch_size]
    else:
        batch_x = temp_x
        batch_y = temp_y
    position += batch_size
    return x_train,y_train,batch_x,batch_y,position

class MnistNet(tcn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = tcn.Linear(784,64)
        self.fc2 = tcn.Linear(64,64)
        self.fc3 = tcn.Linear(64,64)
        self.classfier = tcn.Linear(64,10)

    def forward(self, input):
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.softmax(self.classfier(out))
        return out
    def getloss(self,predict,normal):
        return tcn.CrossEntropyLoss()(predict,normal)
    def getacc(self,predict,normal):
        return np.mean((np.argmax(predict.detach().numpy(),axis=-1)==normal.detach().numpy()))
def train(train_x,train_y,test_x,test_y,train_num,learning_rate,batch_size):
    if isinstance(train_x,tc.Tensor):
        train_x = train_x.numpy().astype(np.float32)
        train_x = np.reshape(train_x,[train_x.shape[0],-1])
    if isinstance(train_y,tc.Tensor):
        train_y = train_y.numpy().astype(np.long)
    if isinstance(test_x,tc.Tensor):
        test_x = test_x.numpy().astype(np.float32)
        test_x = np.reshape(test_x, [test_x.shape[0], -1])
    if isinstance(test_y,tc.Tensor):
        test_y = test_y.numpy().astype(np.long)
    net = MnistNet()
    optimizer = tcoptim.Adam(net.parameters(),lr=learning_rate)
    # step = 0
    for i in range(train_num):
        position = 0
        while(position<train_x.shape[0]):
            train_x, train_y, batch_x, batch_y, position = next_batch(train_x,train_y,position,batch_size)
            predict = net(tc.tensor(batch_x))
            loss = net.getloss(predict,tc.tensor(batch_y))
            acc = net.getacc(predict,tc.tensor(batch_y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t_predict = net(tc.tensor(test_x))
        t_loss = net.getloss(t_predict,tc.tensor(test_y))
        t_acc = net.getacc(t_predict,tc.tensor(test_y))
        print("i=",i,",train-loss=",loss.detach().numpy(),",train-acc=",acc,",test-loss=",t_loss.detach().numpy(),
              "test-acc=",t_acc)


def main():

    tr = tcv.datasets.MNIST("../data/mnist/pytorchversion/",train=True,download=True)
    ts = tcv.datasets.MNIST("../data/mnist/pytorchversion/",train=False)
    train_x,train_y = tr.data,tr.targets
    test_x,test_y = ts.data,ts.targets
    print(train_x.dtype)
    train(train_x,train_y,test_x,test_y,train_num=100,learning_rate=0.0005,batch_size=128)

if __name__ == '__main__':
    main()