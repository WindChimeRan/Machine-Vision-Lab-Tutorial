import numpy as np

# (a ^ b) & c
train_x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
    [0,0,0],
    [0,1,0],
    [1,0,0],
    [1,1,0]
])
train_y = np.array([
    [0],
    [1],
    [1],
    [0],
    [0],
    [0],
    [0],
    [0]
])

def _sigmoid(x,der = False):
    
    if der == True:
        return x*(1-x)

    return 1/(1+np.exp(-x))

def _tanh(x,der = False):
    
    if der == True:
        return 1-(x*x)

    return np.tanh(x)

def _relu(x,der = False):
    
    if der ==True:
        return 1*(x>0)
    
    return x*(x>0)

activation = _sigmoid 

w0 = 2*np.random.random((4,4))-1
w1 = 2*np.random.random((4,1))-1
parameter = (w0,w1)

# cross-entropy

def cross_entropy_train():
    
    i = 3+1
    o = 1
    
    # hyperparameter

    num = 50000
    h = 4
    batch_size = 8
    
    # add_b
    x_with_b = np.column_stack((train_x,np.ones(batch_size)))
    
    #parameter
    w0 = 2*np.random.random((i,h))-1
    w1 = 2*np.random.random((h,o))-1
    

    for epoch in range(num):

        # feed forward
        l0 = x_with_b                           # (batch_size,i)
        l1 = activation(l0.dot(w0))             # (batch_size,h)
        l2 = activation(l1.dot(w1))             # (batch_size,o)

        #loss
        loss = train_y*np.log(l2) + (1-train_y)*np.log(1-l2)
        
        # back propagate
        # l2_err = (train_y - l2)/(l2*(1-l2))        # (batch_size,o)
        # l2_delta = l2_err*activation(l2,der=True)  # (batch_size,o)
        
        l2_delta = train_y - l2
        
        l1_err = l2_delta.dot(w1.T)                # (batch_size,h)
        l1_delta = l1_err*activation(l1,der=True)  # (batch_size,h)
        
        

        ## update parameter
        w1 += l1.T.dot(l2_delta)
        w0 += l0.T.dot(l1_delta)
        
        if epoch%1000 == 0:
            print(np.mean(loss))

        parameter = (w0,w1)
        
    return parameter

inference(cross_entropy_train(),[0,1,1])
