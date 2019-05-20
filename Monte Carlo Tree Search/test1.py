from problem1 import *
import numpy as np
import sys

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (30 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_R_play():
    '''(2 points) Player Random play()'''
    p = PlayerRandom()
    s=np.array([[ 0, 1, 1],
                [ 1, 0,-1],
                [ 1, 1, 0]])
    count=np.zeros(3)
    for _ in range(100):
        r,c = p.play(s)
        assert s[r,c]==0 
        assert r==c 
        assert r>-1 and r<3
        count[c]+=1
    assert count[0]>20
    assert count[1]>20
    assert count[2]>20
    
    s=np.array([[ 1, 1, 0],
                [ 1, 0,-1],
                [ 0, 1, 1]])

    for _ in range(100):
        r,c = p.play(s)
        assert s[r,c]==0 
        assert r==2-c 
        assert r>-1 and r<3
 

#-------------------------------------------------------------------------
def test_T_play_x():
    '''(2 points) TicTacToe play_x()'''
    g = TicTacToe()
    g.play_x(0,0) 
    assert np.allclose(g.s[0,0],1)
    assert np.allclose(g.s.sum(),1)
    g.play_x(2,1) 
    assert np.allclose(g.s[2,1],1)
    assert np.allclose(g.s.sum(),2)


#-------------------------------------------------------------------------
def test_T_play_o():
    '''(2 points) TicTacToe play_o()'''
    g = TicTacToe()
    g.play_o(0,0) 
    assert np.allclose(g.s[0,0],-1)
    assert np.allclose(g.s.sum(),-1)
    g.play_o(2,2) 
    assert np.allclose(g.s[2,2],-1)
    assert np.allclose(g.s.sum(),-2)


#-------------------------------------------------------------------------
def test_T_check():
    '''(3 points) TicTacToe check()'''
    g = TicTacToe()
    e = g.check(g.s)
    assert e is None 
    g.play_x(0,0) 
    g.play_x(0,1) 
    g.play_x(0,2) 
    e = g.check(g.s)
    assert e == 1 
    
    g = TicTacToe()
    g.play_o(0,0) 
    g.play_o(1,0) 
    g.play_o(2,0) 
    e = g.check(g.s)
    assert e == -1
    
    g = TicTacToe()
    g.play_o(0,2) 
    g.play_o(1,1) 
    g.play_o(2,0) 
    e = g.check(g.s)
    assert e == -1
    
    g = TicTacToe()
    g.play_o(2,2) 
    g.play_o(1,1) 
    g.play_o(0,0) 
    e = g.check(g.s)
    assert e == -1

    g = TicTacToe()
    g.s=np.array([[-1, 1,-1],
                  [-1, 1,-1],
                  [ 1,-1, 1]])
    e = g.check(g.s)
    assert e == 0 

    g.s=np.array([[-1, 0,-1],
                  [-1, 1,-1],
                  [ 1,-1, 1]])
    e = g.check(g.s)
    assert e is None 


#-------------------------------------------------------------------------
def test_T_game():
    '''(5 points) TicTacToe game()'''

    # if the game has already ended
    g = TicTacToe()
    p1 = PlayerRandom()
    g.s=np.array([[ 0, 1, 1],
                  [-1,-1,-1],
                  [ 1,-1, 1]])
    e = g.game(p1,p1)
    assert e==-1



    p1 = PlayerRandom()
    p2 = PlayerRandom()
    w =0  
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0,-1, 1],
                      [-1, 1, 0],
                      [-1, 1,-1]])
        e = g.game(p1,p2)
        w+=e
    print(w)
    assert w<0
    assert w<-30
    assert w>-70

    class test1:
        def play(self,s):
            assert s[1,1] == 1
            assert s[2,2] ==-1
            r,c=np.where(s==0)
            return r[0],c[0]
    class test2:
        def play(self,s):
            assert s[1,1] ==-1
            assert s[2,2] == 1
            r,c=np.where(s==0)
            return r[0],c[0]


    g = TicTacToe()

    p1 = test1()
    p2 = test2()
    
    g.s=np.array([[ 0, 0, 0],
                  [ 0, 1, 0],
                  [ 0, 0,-1]])
     
    e = g.game(p1,p2)


    class test3:
        def play(self,s):
            r,c=np.where(s==0)
            return r[0],c[0]


    g = TicTacToe()

    p1 = test3()
    e = g.game(p1,p1)
    print(g.s)
    s=np.array([[ 1,-1, 1],
                [-1, 1,-1],
                [ 1, 0, 0]])
    assert np.allclose(g.s, s)
    assert e==1


#-------------------------------------------------------------------------
def test_M_update_v():
    '''(2 points) Player MiniMax update_v()'''
    p = PlayerMiniMax()
    s=np.array([[ 1, 0, 0],
                [ 0, 1, 0],
                [ 0,-1, 1]])
    p.update_v(s,1) # won the game
    e = p.v[str(s)]
    assert  e== 1
    assert len(p.v.keys())== 1



#-------------------------------------------------------------------------
def test_M_update_p():
    '''(2 points) Player MiniMax update_p()'''
    p = PlayerMiniMax()
    s=np.array([[ 0, 0, 0],
                [ 0, 1, 0],
                [ 0, 1,-1]])
    p.update_p(s,0,1) 
    r,c = p.p[str(s)]
    assert  r== 0
    assert  c== 1
    assert len(p.p.keys())== 1



#-------------------------------------------------------------------------
def test_M_compute_v():
    '''(5 points) Player MiniMax compute_v()'''
    p = PlayerMiniMax()
    s=np.array([[ 1, 0, 0],
                [ 0, 1, 0],
                [ 0,-1, 1]])
    v=p.compute_v(s) # won the game
    assert  v== 1

    e = p.v[str(s)]
    assert  e== 1

    assert len(p.v.keys())== 1


    s=np.array([[ 0, 0, 0],
                [-1,-1,-1],
                [ 0, 1, 0]])
    v=p.compute_v(s) 
    assert  v==-1

    s=np.array([[-1, 1,-1],
                [-1, 1,-1],
                [ 1,-1, 1]])
    v=p.compute_v(s) 
    assert  v==0

    s=np.array([[-1, 1,-1],
                [-1, 1,-1],
                [ 0,-1, 1]])
    v=p.compute_v(s) 
    assert  v==0


    s=np.array([[-1,-1, 1],
                [-1, 1,-1],
                [ 0,-1, 1]])
    v=p.compute_v(s) 
    assert  v==1


    s=np.array([[ 0, 1,-1],
                [-1,-1, 1],
                [ 0,-1, 1]])
    v=p.compute_v(s) 
    assert v==0  


    s=np.array([[ 0, 1, 1],
                [-1, 1,-1],
                [ 1,-1, 0]])
    v=p.compute_v(s) 
    assert v==1  

    s=np.array([[ 0, 0, 1],
                [-1, 1, 0],
                [-1, 0, 0]])
    v=p.compute_v(s) 
    assert v==1  


#-------------------------------------------------------------------------
def test_M_play():
    '''(2 points) Player MiniMax play()'''
    p = PlayerMiniMax()
    s=np.array([[-1, 1,-1],
                [-1, 1,-1],
                [ 0,-1, 1]])
    r, c = p.play(s)
    assert np.allclose(s,[[-1, 1,-1], [-1, 1,-1], [ 0,-1, 1]])
    assert r==2  
    assert c==0  


    s=np.array([[-1,-1, 1],
                [-1, 1,-1],
                [ 0,-1, 1]])
    r, c = p.play(s)
    assert r==2
    assert c==0  

    p = PlayerMiniMax()
    s=np.array([[ 0,-1, 1],
                [-1, 1,-1],
                [ 0,-1,-1]])
    r, c = p.play(s)
    assert r==2  
    assert c==0  

    p = PlayerMiniMax()
    s=np.array([[ 0, 1,-1],
                [-1,-1, 1],
                [ 0,-1, 1]])
    r, c = p.play(s)
    assert r==2  
    assert c==0  

    p = PlayerMiniMax()
    s=np.array([[ 0, 1, 1],
                [-1, 1,-1],
                [-1,-1, 1]])
    r, c = p.play(s)
    assert r==0  
    assert c==0  

    p = PlayerMiniMax()
    s=np.array([[ 0,-1, 1],
                [-1, 1,-1],
                [-1, 1, 0]])
    r, c = p.play(s)
    assert r==0  
    assert c==0  




#-------------------------------------------------------------------------
def test_players():
    '''(2 points) random vs Minimax'''

    p1 = PlayerMiniMax()
    p2 = PlayerRandom()
    w=0
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0,-1, 1],
                      [-1, 1,-1],
                      [ 0,-1,-1]])
        e = g.game(p1,p2)
        w += e
    assert w==100

    w=0
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0,-1, 1],
                      [-1, 1,-1],
                      [-1, 1, 0]])
        e = g.game(p1,p2)
        w += e
    assert w==0


    w=0
    p1 = PlayerMiniMax()
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0, 0, 1],
                      [ 0,-1, 0],
                      [ 1,-1, 0]])
        e = g.game(p1,p2)
        w += e
    assert np.abs(w-87)<10

#-------------------------------------------------------------------------
def test_players2():
    '''(3 points) Minimax vs Minimax'''
    # NOTE: this test can usually finish within 20 seconds. 
    # if your code is very slow (say using more than 1 minute), you may want to check the dictionaries (self.v and self.p) implementation.

    p = PlayerMiniMax()
    w=0
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0, 0, 1],
                      [ 0,-1, 0],
                      [ 1,-1, 0]])
        e = g.game(p,p)
        w += e
    assert w==0

    w=0
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0, 0, 0],
                      [ 0,-1, 0],
                      [ 1, 0, 0]])
        e = g.game(p,p)
        w += e
    assert w==0

    w=0
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0, 0, 0],
                      [ 0, 0, 0],
                      [ 1,-1, 0]])
        e = g.game(p,p)
        w += e
    assert w==100

    w=0
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0, 0, 0],
                      [ 0, 1, 0],
                      [ 0,-1, 0]])
        e = g.game(p,p)
        w += e
    assert w==100

    w=0
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0, 0, 0],
                      [ 0, 1, 0],
                      [-1, 0, 0]])
        e = g.game(p,p)
        w += e
    assert w==0

    w=0
    for i in range(100):
        g = TicTacToe()
        e = g.game(p,p)
        w += e
    assert w==0


