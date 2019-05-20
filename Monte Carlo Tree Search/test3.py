from problem3 import *
import numpy as np
import sys

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (40 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_expand():
    '''(5 points) expand'''
    s=np.array([[ 0,-1,-1],
                [ 0, 1, 1],
                [-1, 1,-1]])
    n = Node(s)
    PlayerMCTS.expand(n)
    assert n.isleaf==False
    assert len(n.children) ==2 
    s_=np.array([[ 0,-1,-1],
                 [ 0, 1, 1],
                 [-1, 1,-1]])
    assert np.allclose(n.s,s_)
    for x in n.children:
        assert x.parent == n
    assert n.N==0
    assert n.w==0

    s=np.array([[ 0, 1, 1],
                [-1,-1,-1],
                [ 1,-1, 1]])
    c = False
    for x in n.children:
        if np.allclose(x.s,s):
            c=True
    assert c

    s=np.array([[-1, 1, 1],
                [ 0,-1,-1],
                [ 1,-1, 1]])
    c = False
    for x in n.children:
        if np.allclose(x.s,s):
            c=True
    assert c



    s=np.array([[ 0,-1,-1],
                [-1, 1, 1],
                [-1, 1,-1]])
    n = Node(s)
    PlayerMCTS.expand(n)
    assert n.isleaf==False
    assert len(n.children) ==1
    assert n.children[0].isleaf ==True


    s=np.array([[ 0,-1,-1],
                [ 1, 1, 1],
                [-1, 1,-1]])
    n = Node(s)
    PlayerMCTS.expand(n)
    assert n.isleaf==True 
    assert len(n.children) ==0

    



#-------------------------------------------------------------------------
def test_rollout():
    '''(5 points) rollout'''

    p = PlayerMCTS()
    s=np.array([[ 0,-1,-1],
                [ 0, 1, 1],
                [-1, 1,-1]])
    w = 0
    for _ in range(1000):
        w += p.rollout(s)
    
    s_=np.array([[ 0,-1,-1],
                 [ 0, 1, 1],
                 [-1, 1,-1]])
    assert np.allclose(s,s_)
    assert np.abs(w-500) <100

    s=np.array([[ 0, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, 0]])
    w = 0
    for _ in range(1000):
        w += p.rollout(s)
    assert np.abs(w-800) <100


    s=np.array([[ 0, 0, 0],
                [ 0,-1, 0],
                [ 0, 0, 0]])
    w = 0
    for _ in range(1000):
        w += p.rollout(s)
    assert np.abs(w+500) <100



#-------------------------------------------------------------------------
def test_backprop():
    '''(5 points) backprop'''
    s=np.array([[ 0,-1,-1],
                [ 0, 1, 1],
                [-1, 1,-1]])
    n = Node(s)
    PlayerMCTS.expand(n)
    c = n.children[0]
    PlayerMCTS.backprop(c,-1)
    assert c.w ==-1
    assert c.N ==1
    assert n.w ==1
    assert n.N ==1

    PlayerMCTS.expand(c)
    cc = c.children[0]
    PlayerMCTS.backprop(cc,1)
    assert cc.w ==1
    assert cc.N ==1
    assert c.w ==-2
    assert c.N ==2
    assert n.w ==2
    assert n.N ==2

    PlayerMCTS.backprop(c,1)
    assert c.w ==-1
    assert c.N ==3
    assert n.w ==1
    assert n.N ==3
    

#-------------------------------------------------------------------------
def test_selection():
    '''(5 points) selection'''

    #leaf node
    s=np.array([[ 0,-1,-1],
                [ 1, 1, 1],
                [-1, 1,-1]])
    n = Node(s)
    n.isleaf == True
    node = PlayerMCTS.selection(n)
    assert node ==n

    # unexpanded
    s=np.array([[ 0,-1,-1],
                [ 0, 1, 1],
                [-1, 1,-1]])
    n = Node(s)
    node = PlayerMCTS.selection(n)
    assert node ==n

    # UCB 
    s=np.array([[ 0,-1,-1],
                [ 0, 1, 1],
                [-1, 1,-1]])
    n = Node(s)
    PlayerMCTS.expand(n)
    c1 = n.children[0]
    PlayerMCTS.backprop(c1,-1)

    c2 = n.children[1]
    node = PlayerMCTS.selection(n)
    assert node ==c2



    # recursion
    s=np.array([[ 0,-1, 1],
                [ 0, 1,-1],
                [ 0, 1,-1]])
    n = Node(s)
    PlayerMCTS.expand(n)
    c1 = n.children[0]
    c2 = n.children[1]
    c3 = n.children[2]
    PlayerMCTS.backprop(c1,-1)
    PlayerMCTS.backprop(c2,1)
    PlayerMCTS.backprop(c3,-1)
    node = PlayerMCTS.selection(n)
    assert node ==c2
    
    PlayerMCTS.expand(c2)
    assert len(c2.children)==2
    c2c1=c2.children[0]
    c2c2=c2.children[1]
    PlayerMCTS.backprop(c2c1, 1)
    
    node = PlayerMCTS.selection(n)
    assert node ==c2c2

    PlayerMCTS.backprop(c2c2,-1)
    node = PlayerMCTS.selection(n)
    assert node ==c2c1
   



    n = Node(s)
    PlayerMCTS.expand(n)
    c1 = n.children[0]
    c2 = n.children[1]
    c3 = n.children[2]
    PlayerMCTS.backprop(c1, 0)
    PlayerMCTS.backprop(c2,1)
    PlayerMCTS.backprop(c3, -1)
    node = PlayerMCTS.selection(n)
    assert node ==c2
    
    PlayerMCTS.expand(c2)
    c2c1=c2.children[0]
    c2c2=c2.children[1]
    PlayerMCTS.backprop(c2c1, 1)
    PlayerMCTS.backprop(c2c1, 1)
    
    node = PlayerMCTS.selection(n)
    assert node ==c1





#-------------------------------------------------------------------------
def test_build_tree():
    '''(5 points) build_tree'''

    s=np.array([[ 0,-1,-1],
                [ 0, 1, 1],
                [ 1, 1,-1]])
    r = PlayerMCTS.build_tree(s,2)
    print(r.children[0].s)
    print(r.children[1].s)
    assert r.children[0].N==1
    assert r.children[1].N==1

    assert r.children[0].w==0
    assert r.children[1].w==1


    r = PlayerMCTS.build_tree(s,3)
    assert r.children[0].N==1
    assert r.children[1].N==2
    assert r.children[0].w==0
    assert r.children[1].w==2


    r = PlayerMCTS.build_tree(s,8)
    assert r.children[0].N==1
    assert r.children[1].N==7
    assert r.children[0].w==0
    assert r.children[1].w==7

    r = PlayerMCTS.build_tree(s,9)
    assert r.children[0].N==2
    assert r.children[1].N==7
    assert r.children[0].w==0
    assert r.children[1].w==7


    s=np.array([[ 0,-1, 1],
                [ 0,-1, 1],
                [ 0, 1,-1]])
    r = PlayerMCTS.build_tree(s)
    assert r.children[0].N > 50
    assert r.children[0].w == 0 

    s=np.array([[ 0,-1,-1],
                [ 0, 1, 0],
                [ 0, 0, 0]])
    r = PlayerMCTS.build_tree(s)
    assert r.children[0].N >50 
    assert r.children[0].w < 20 

    c1= r.children[0]
    c1c = c1.children[-1]
    print(c1c.N)
    print(c1.N)
    assert c1c.N >= c1.N/2 -1 
    assert c1c.w < 10






#-------------------------------------------------------------------------
def test_play():
    '''(5 points) play'''
    p =PlayerMCTS()
    s=np.array([[ 0,-1,-1],
                [ 0, 1, 0],
                [ 0, 0, 0]])
     
    r,c=p.play(s)
    assert r ==0
    assert c ==0

    s=np.array([[ 0, 0,-1],
                [ 0, 1,-1],
                [ 0, 0, 0]])
     
    r,c=p.play(s)
    assert r ==2
    assert c ==2



#-------------------------------------------------------------------------
def test_players():
    '''(5 points) random vs MCTS'''

    p1 = PlayerMCTS()
    p2 = PlayerRandom()
    w=0
    for i in range(10):
        g = TicTacToe()
        g.s=np.array([[ 0,-1, 1],
                      [-1, 1,-1],
                      [ 0,-1,-1]])
        e = g.game(p1,p2)
        w += e
    assert w==10

    w=0
    for i in range(10):
        g = TicTacToe()
        g.s=np.array([[ 0,-1, 1],
                      [-1, 1,-1],
                      [-1, 1, 0]])
        e = g.game(p1,p2)
        w += e
    assert w==0


#-------------------------------------------------------------------------
def test_players2():
    '''(5 points) Minimax vs MCTS '''

    p1 = PlayerMCTS()
    p2 = PlayerMiniMax()
    w=0
    for i in range(10):
        g = TicTacToe()
        g.s=np.array([[ 0, 0, 1],
                      [ 0,-1, 0],
                      [ 1,-1, 0]])
        e = g.game(p1,p2)
        w += e
    assert w==0


    w=0
    for i in range(10):
        g = TicTacToe()
        g.s=np.array([[ 0, 0, 0],
                      [ 0, 0, 0],
                      [ 1,-1, 0]])
        e = g.game(p1,p2)
        w += e
    print(w)
    assert w>5
