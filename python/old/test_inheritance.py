class A:
    def __init__(self):
        self.say_something()

    def say_something(self):
        print 'class A'

class B(A):

    def say_something(self):
        print "class B"

class C(A):
    def __init__(self,phrase='hello C'):
        self.phrase = phrase
        A.__init__(self)



    def say_something(self):
        print self.phrase

b = B()

c = C()
