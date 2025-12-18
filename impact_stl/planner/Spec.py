import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utilities.zonotopes import zonotope
class Robot():
    # Robot class
    def __init__(self,name: str ,x0: np.ndarray ,dx0:np.ndarray ,xf: np.ndarray,dxf: np.ndarray,nbz: int =10,
                 dq_lb: np.ndarray = np.array([-2,-2]), dq_ub: np.ndarray = np.array([2,2])):
        """
        The robot is controlled by a sequence of bezier curves, each with its own
        lower and upper bound on velocity.
        The bezier curves can be of three types: 'none', 'pre' and 'post' impact.
        nbz is the number of bezier curves used for the entire trajectory.
        Adding more allows for more complex trajectories, but increases the number of variables and computation time.
        Args:
            name: name of the robot
            x0: initial position (np.array([x,y]))
            dx0: initial velocity (np.array([dx,dy]))
            xf: final position (np.array([x,y]))
            dxf: final velocity (np.array([dx,dy]))
            nbz: number of bezier curves
            dq_lb: lower bound on velocity (np.array([dx_min,dy_min]))
            dq_ub: upper bound on velocity (np.array([dx_max,dy_max]))

        """
        assert x0 is not None, "Robot must have an initial position"
        assert dx0 is not None, "Robot must have an initial velocity"

        
        self.name = name
        self.mass = 16.8
        self.x0 = x0
        self.dx0 = dx0
        self.xf = xf
        self.dxf = dxf

        self.nbz = nbz
        self.dq_lb = dq_lb
        self.dq_ub = dq_ub

        # the kind of bezier curve: ['none','pre','post']-impact
        self.ids = self.nbz*['none']
        # other names contains the name of the robot or object
        # that the object or robot is impacting with, only if 
        # self.ids[i] == 'pre' or 'post'!
        self.other_names = self.nbz*['none']


class Object():
    def __init__(self, name: str, x0: np.ndarray, dx0: np.ndarray, xf: np.ndarray, dxf: np.ndarray, nbz: int=10, t0: int=0, tf: int=100,
                 dq_lb: np.ndarray=np.array([-2,-2]), dq_ub:np.ndarray=np.array([2,2])):
        """
        Object must have a initial velocity and position, but not necessarily a final position or velocity.
        Args:
            name: name of the object
            x0: initial position (np.array([x,y]))
            dx0: initial velocity (np.array([dx,dy]))
            xf: final position (np.array([x,y])) or None if unknown
            dxf: final velocity (np.array([dx,dy]))
            nbz: number of bezier curves
            t0: initial time
            tf: final time
            dq_lb: lower bound on velocity (np.array([dx_min,dy_min]))
            dq_ub: upper bound on velocity (np.array([dx_max,dy_max]))
        """
        assert x0 is not None, "Object must have an initial position"
        assert dx0 is not None, "Object must have an initial velocity"
        self.name = name
        self.mass = 16.8
        self.x0 = x0
        self.dx0 = dx0
        self.xf = xf
        self.dxf = dxf

        self.Xfd = zonotope(x=np.append(xf,dxf),
                            Gdiag=np.array([0.5,0.5,0.05,0.05]))

        # create the initial zonotope set, without uncertainty on velocity
        self.X0d = zonotope(x=np.append(x0,dx0),
                            G=np.array([[0,0,0,0],
                                        [0,0,0,0],
                                        [0,0,0,0],
                                        [0,0,0,0]]))
        # array to keep track of Xfs for step
        self.Xfs = []

        self.nbz = nbz
        self.dt = (tf-t0)/self.nbz
        #hvar is a list of nparrays of time intervals for each bezier curve
        self.hvar = [np.array([[t0+i*self.dt,t0+(i+1)*self.dt]]) for i in range(self.nbz)]
        # velocity bounds
        self.dq_lb = dq_lb
        self.dq_ub = dq_ub

        # the kind of bezier curve: ['none','pre','post']-impact
        self.ids = self.nbz*['none']
        # other names contains the name of the robot or object
        # that the object or robot is impacting with, only if
            # self.ids[i] == 'pre' or 'post'!
        self.other_names = self.nbz*['none']

    def evaluate_t(self,t: float):
        """
        Given a time t, return the index of the bezier curve that contains t
        and the phase between [0,1] of the bezier curve
        
        Args:
            t: time
            Returns:
            idx: index of the bezier curve that contains t
            phase: phase between [0,1] of the bezier curve
            
            
        """
        # return the idx of the hvar that contains t
        # also return the phase between [0,1]
        for i,h in enumerate(self.hvar):
            if t >= h[0] and t < h[1]:
                return i, (t-h[0])/(h[1]-h[0])
            
        return i,1


class Area():
    def __init__(self,x_min: np.ndarray, x_max:np.ndarray) -> None:
        """
        Axis-aligned rectangular area defined by its minimum and maximum corners.
        Args:
            x_min: minimum corner (np.array([x_min,y_min]))
            x_max: maximum corner (np.array([x_max,y_max]))
        """
        self.x_min = x_min
        self.x_max = x_max
        # and convert x_min and x_max to a polytope, Hx <= b
        self.H = np.array([[1,0],
                           [0,1],
                           [-1,0],
                           [0,-1]])
        self.b = np.array([x_max[0],x_max[1],-x_min[0],-x_min[1]])
        self.nfaces = 4

    def plot(self,ax,color='r') -> None:
        """
        Plot the area on the given axis
        Args:
            ax: matplotlib axis
            color: color of the area
        """
        # draw a red rectangle
        rect = Rectangle((self.x_min[0],self.x_min[1]),self.x_max[0]-self.x_min[0],self.x_max[1]-self.x_min[1],
                         linewidth=1,edgecolor=color,facecolor=color)
        ax.add_patch(rect)


class Pred():
    def __init__(self,type: str, I: np.ndarray=[0,0], preds: list=[],io: str="in") -> None:
        """
        STL Predicate class
        Args:
            type: type of the predicate, one of ['MU','NEG','AND','OR','NOT','F','G','U']
            I: time interval for temporal operators, e.g. [t1,t2] for F_[t1,t2]
            preds: list of sub-predicates
            io: "in" or "out", only for type "MU" and "NEG"

            For type "MU" and "NEG", preds is an Area object
            For obstacle avoidance use Pred(type="NEG",preds=obs1, io = "in")
            For type "AND", "OR", "NOT", preds is a list of Pred objects
        """
        self.type = type
        self.I = I
        self.preds = preds
        self.io = io
        self.rho = None

        self.z_time = None
    
    def get_string(self)-> str:
        """
        Return a string representation of the predicate
        
        """
        return f"{self.type}({self.I})"

class Spec():
    def __init__(self,t0,tf) -> None:

        """
        STL Specification class
        Args:
            t0: initial time
            tf: final time
        
        The specification is a tree structure, where each node is a predicate
            and the leaves are the atomic predicates (e.g. mu, neg, etc.)

        """
        # list of predicates
        self.preds = []
        # list of names
        self.names = []
        # root predicate
        self.pred = None

        self.t0 = t0
        self.tf = tf

    def add_pred(self,pred,name):
        # make deep copy of pred
        self.names.append(copy.deepcopy(name))
        self.preds.append(copy.deepcopy(pred))
        print(f"Added pred {pred.get_string()} with name {name}")
        print(f"names: {self.names}")

def spatial_specifications(world: Object ,specification: str) -> None:
    """
    Define spatial STL specifications for the given world.
    Preds are standard but, final velocity, position and number of bezier curves need to be considered for the feasibility of the problem.

    """

    if specification == "obstacle_avoidance_arian":
        # STL 
        # This is a bit wonky, but it shows what is possible. Although there are several times the robot and object cross paths
        # without impacting.
        area1 = Area(x_min=np.array([2,2]),x_max=np.array([3,3]))
        mu1 = Pred(type="MU",preds=area1,io="in")
        phi1 = Pred(type="F",I=[15,20],preds=[mu1])
        area2 = Area(x_min=np.array([8,8]),x_max=np.array([9,9]))
        mu2 = Pred(type="MU",preds=area2,io="in")
        phi2 = Pred(type="F",I=[50,100],preds=[mu2])

        obs1 = Area(x_min=np.array([5,2]),x_max=np.array([7,4]))
        phi31 = Pred(type="NEG",preds=obs1) 
        phi3 = Pred(type="G",I=[0,100],preds=[phi31])

        world.spec = Spec(t0=0,tf=100)
        world.spec.add_pred(copy.deepcopy(phi3),
                           name='crackle')
        world.spec.add_pred(copy.deepcopy(phi3),
                           name='snap')
        world.spec.add_pred(Pred(type="AND",preds=[phi1,phi2,phi3]),
                         name='pop')

        robot1 = Robot(name="snap",
                       x0=np.array([9,5]),
                       dx0=np.array([0,0]),
                       xf=np.array([5,5]),
                       dxf=np.array([0,0]),nbz=6)
        
        robot2 = Robot(name="crackle",
                       x0=np.array([1,7]),
                       dx0=np.array([0,0]),
                       xf=np.array([5,5]),
                       dxf=np.array([0,0]),nbz=6)
        
        object1 = Object(name="pop",
                         x0=np.array([5.5,5]),
                         dx0=np.array([0,0]),
                         xf=np.array([8.5,8.5]),
                         dxf=np.array([0,0]),nbz=6)

        world.dim = 2
        world.robots = [robot1,robot2]
        world.objects = [object1]

        # World bounding box
        world.x_lb = np.array([0,0])
        world.x_ub = np.array([10,10])

        # Obstacles
        world.obstacles= [obs1]

        # Area's of interest
        world.areas = [area1,area2]

    elif specification == "minimal_test_obstacle":
        tf = 150

        #just so I can mark the final location of the object, not actually used in an stl spec
        area1 = Area(x_min=np.array([4.5,24.5]),x_max=np.array([5.5,25.5]))



        obs1 = Area(x_min=np.array([3,15]),x_max=np.array([8,20]))
        phi31 = Pred(type="NEG",preds=obs1) 
        phi3 = Pred(type="G",I=[0,tf],preds=[phi31])
        
        world.spec = Spec(t0=0,tf=tf)
        world.spec.add_pred(phi3, name='crockle')
        world.spec.add_pred(phi3, name='snap')
        world.spec.add_pred(phi3, name='pop')
        
        bz = 15
        robot1 = Robot(name="snap",
                       x0=np.array([1,4]),
                       dx0=np.array([0,0]),
                       xf=None,
                       dxf=None,nbz=bz)
        robot2 = Robot(name="crockle",
                       x0=np.array([2,9]),
                       dx0=np.array([0,0]),
                       xf=None,
                       dxf=None,nbz=bz)
        
        object1 = Object(name="pop",
                         x0=np.array([3,3]),
                         dx0=np.array([0,0]),
                         xf=np.array([5,25]),
                         dxf=np.array([0,0]),nbz=bz)

        world.dim = 2
        world.robots = [robot1,robot2]
        world.objects = [object1]

        # World bounding box
        world.x_lb = np.array([0,0])
        world.x_ub = np.array([30,30])

        # Obstacles
        world.obstacles= [obs1]

        # Area's of interest
        world.areas = [area1]

    elif specification == "test4":
        tf = 90

        #just so I can mark the final location of the object, not actually used in an stl spec
        area1 = Area(x_min=np.array([24.5,24.5]),x_max=np.array([25.5,25.5]))



        obs1 = Area(x_min=np.array([12,12]),x_max=np.array([18,18]))
        phi31 = Pred(type="NEG",preds=obs1) 
        phi3 = Pred(type="G",I=[0,tf],preds=[phi31])
        
        world.spec = Spec(t0=0,tf=tf)
        world.spec.add_pred(phi3, name='crackle')
        world.spec.add_pred(phi3, name='snap')
        world.spec.add_pred(phi3, name='pop')
        
        bz = 8
        robot1 = Robot(name="snap",
                       x0=np.array([1,4]),
                       dx0=np.array([0,0]),
                       xf=None,
                       dxf=None,nbz=bz)
        robot2 = Robot(name="crackle",
                       x0=np.array([2,9]),
                       dx0=np.array([0,0]),
                       xf=None,
                       dxf=None,nbz=bz)
        
        object1 = Object(name="pop",
                         x0=np.array([3,3]),
                         dx0=np.array([0,0]),
                         xf=np.array([25,25]),
                         dxf=np.array([0,0]),nbz=2*bz)

        world.dim = 2
        world.robots = [robot1,robot2]
        world.objects = [object1]

        # World bounding box
        world.x_lb = np.array([0,0])
        world.x_ub = np.array([30,30])

        # Obstacles
        world.obstacles= [obs1]

        # Area's of interest
        world.areas = [area1]


    elif specification == "test1":
        tf = 50

        world.spec = Spec(t0=0,tf=tf)
        bz = 6
        
        robot1 = Robot(name="snap",
                       x0=np.array([5,0]),
                       dx0=np.array([0,0]),
                       xf=np.array([5,0]),
                       dxf=np.array([0,0]),nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        
        robot2 = Robot(name="crackle",
                       x0=np.array([5,20]),
                       dx0=np.array([0,0]),
                       xf=np.array([5,20]),
                       dxf=np.array([0,0]),nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        
        object1 = Object(name="pop",
                         x0=np.array([5,2]),
                         dx0=np.array([0,0]),
                         xf=np.array([5,18]),
                         dxf=np.array([0,0]),nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        
        area1 = Area(x_min=np.array([4.5,17.5]),x_max=np.array([5.5,18.5]))

        world.dim = 2
        world.robots = [robot1,robot2]
        world.objects = [object1]

        # World bounding box
        world.x_lb = np.array([0,0])
        world.x_ub = np.array([10,20])

        # Obstacles
        world.obstacles= []

        # Area's of interest
        world.areas = [area1]
    elif specification == "test2":
        tf = 50
        world.spec = Spec(t0=0,tf=tf)

        bz = 6
        
        robot1 = Robot(name="snap",
                       x0=np.array([2,2]),
                       dx0=np.array([0,0]),
                       xf=None,
                       dxf=None,nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        
        robot2 = Robot(name="crackle",
                       x0=np.array([20,20]),
                       dx0=np.array([0,0]),
                       xf=None,
                       dxf=None,nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        
        object1 = Object(name="pop",
                         x0=np.array([4,4]),
                         dx0=np.array([0,0]),
                         xf=np.array([25,25]),
                         dxf=np.array([0,0]),nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        
        area1 = Area(x_min=np.array([24.5,24.5]),x_max=np.array([25.5,25.5]))

        world.dim = 2
        world.robots = [robot1,robot2]
        world.objects = [object1]

        # World bounding box
        world.x_lb = np.array([0,0])
        world.x_ub = np.array([30,30])

        # Obstacles
        world.obstacles= []

        # Area's of interest
        world.areas = [area1]
    elif specification == "test3":
        tf = 100

        obs1 = Area(x_min=np.array([0,10]),x_max=np.array([18,16]))
        phi31 = Pred(type="NEG",preds=obs1) 
        phi3 = Pred(type="G",I=[0,tf],preds=[phi31])
        
        world.spec = Spec(t0=0,tf=tf)
        world.spec.add_pred(phi3, name='crackle')
        world.spec.add_pred(phi3, name='snap')
        world.spec.add_pred(phi3, name='pop')


        bz = 8
        
        robot1 = Robot(name="snap",
                       x0=np.array([2,2]),
                       dx0=np.array([0,0]),
                       xf=None,
                       dxf=None,nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        
        robot2 = Robot(name="crackle",
                       x0=np.array([20,20]),
                       dx0=np.array([0,0]),
                       xf=None,
                       dxf=None,nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        
        object1 = Object(name="pop",
                         x0=np.array([4,4]),
                         dx0=np.array([0,0]),
                         xf=np.array([25,25]),
                         dxf=np.array([0,0]),nbz=bz*2, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        
        area1 = Area(x_min=np.array([24.5,24.5]),x_max=np.array([25.5,25.5]))

        world.dim = 2
        world.robots = [robot1,robot2]
        world.objects = [object1]

        # World bounding box
        world.x_lb = np.array([0,0])
        world.x_ub = np.array([30,30])

        # Obstacles
        world.obstacles= [obs1]

        # Area's of interest
        world.areas = [area1]

    elif specification == "final_test":
        tf = 120

        #just so I can mark the final location of the object, not actually used in an stl spec
        area1 = Area(x_min=np.array([24.5,24.5]),x_max=np.array([25.5,25.5]))



        obs1 = Area(x_min=np.array([12,12]),x_max=np.array([30,18]))
        phi31 = Pred(type="NEG",preds=obs1) 
        phi3 = Pred(type="G",I=[0,tf],preds=[phi31])
        
        world.spec = Spec(t0=0,tf=tf)
        world.spec.add_pred(phi3, name='crackle')
        world.spec.add_pred(phi3, name='snap')
        world.spec.add_pred(phi3, name='pop')
        
        bz = 10
        robot1 = Robot(name="snap",
                       x0=np.array([25,2]),
                       dx0=np.array([0,0]),
                       xf=None,
                       dxf=None,nbz=bz, dq_lb=np.array([-3,-3]),dq_ub=np.array([3,3]))
        robot2 = Robot(name="crackle",
                       x0=np.array([5,15]),
                       dx0=np.array([0,0]),
                       xf=None,
                       dxf=None,nbz=bz, dq_lb=np.array([-3,-3]),dq_ub=np.array([3,3]))
        
        object1 = Object(name="pop",
                         x0=np.array([20,5]),
                         dx0=np.array([0,0]),
                         xf=np.array([25,25]),
                         dxf=np.array([0,0]),nbz=2*bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))

        world.dim = 2
        world.robots = [robot1,robot2]
        world.objects = [object1]

        # World bounding box
        world.x_lb = np.array([0,0])
        world.x_ub = np.array([30,30])

        # Obstacles
        world.obstacles= [obs1]

        # Area's of interest
        world.areas = [area1]

    elif specification == "hw_test":
        tf = 15

        world.spec = Spec(t0=0,tf=tf)
        bz = 2
        
        robot1 = Robot(name="pop",
                       x0=np.array([1,0]),
                       dx0=np.array([0,0]),
                       xf=np.array([2.5,0]),
                       dxf= np.array([0,0]),nbz=bz, dq_lb=np.array([-0.2,-0.2]),dq_ub=np.array([0.2,0.2]))
        
        #robot2 = Robot(name="crackle",
        #               x0=np.array([5,20]),
        #               dx0=np.array([0,0]),
        #               xf=np.array([5,20]),
        #               dxf=np.array([0,0]),nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        #
        #object1 = Object(name="pop",
        #                 x0=np.array([2.5,0]),
        #                 dx0=np.array([0,0]),
        #                 xf=np.array([2.5,0]),
        #                 dxf=None,nbz=bz, dq_lb=np.array([-0.1,-0.1]),dq_ub=np.array([0.1,0.1]))
        
        area1 = Area(x_min=np.array([2.25,-0.25]),x_max=np.array([2.75,0.25]))

        world.dim = 2
        world.robots = [robot1]
        world.objects = []

        # World bounding box
        world.x_lb = np.array([0,-1.75])
        world.x_ub = np.array([3.5,1.75])

        # Obstacles
        world.obstacles= []

        # Area's of interest
        world.areas = [area1]

    elif specification == "hw_test2":
        tf = 22

        world.spec = Spec(t0=0,tf=tf)
        bz = 4
        
        robot1 = Robot(name="pop",
                       x0=np.array([0.5,0]),
                       dx0=np.array([0,0]),
                       xf=np.array([2.0,0]),
                       dxf= None,nbz=bz, dq_lb=np.array([-0.2,-0.2]),dq_ub=np.array([0.2,0.2]))
        
        #robot2 = Robot(name="crackle",
        #               x0=np.array([5,20]),
        #               dx0=np.array([0,0]),
        #               xf=np.array([5,20]),
        #               dxf=np.array([0,0]),nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        #
        object1 = Object(name="crackle",
                         x0=np.array([1.5,0]),
                         dx0=np.array([0,0]),
                         xf=np.array([2.5,0]),
                         dxf=None,nbz=bz, dq_lb=np.array([-0.2,-0.2]),dq_ub=np.array([0.2,0.2]))
        
        area1 = Area(x_min=np.array([2.25,-0.25]),x_max=np.array([2.75,0.25]))

        world.dim = 2
        world.robots = [robot1]
        world.objects = [object1]

        # World bounding box
        world.x_lb = np.array([0,-1.75])
        world.x_ub = np.array([3.5,1.75])

        # Obstacles
        world.obstacles= []

        # Area's of interest
        world.areas = [area1]
    # elif specification == "hw_test3":
    #     tf = 35

    #     world.spec = Spec(t0=0,tf=tf)
    #     bz = 5
        
    #     robot1 = Robot(name="pop",
    #                    x0=np.array([0.5,-1.25]),
    #                    dx0=np.array([0,0]),
    #                    xf=np.array([0.5,-1.25]),
    #                    dxf= np.array([0,0]),nbz=bz, dq_lb=np.array([-0.2,-0.2]),dq_ub=np.array([0.2,0.2]))
        
    #     robot2 = Robot(name="crackle",
    #                   x0=np.array([3.25,1.5]),
    #                   dx0=np.array([0,0]),
    #                   xf=None,
    #                   dxf=None,nbz=bz, dq_lb=np.array([-2,-2]),dq_ub=np.array([2,2]))
        
    #     object1 = Object(name="snap",
    #                      x0=np.array([1.25,-0.50]),
    #                      dx0=np.array([0,0]),
    #                      xf=np.array([2.5,0.75]),
    #                      dxf=np.array([0,0]),nbz=bz, dq_lb=np.array([-0.2,-0.2]),dq_ub=np.array([0.2,0.2]))
        
    #     area1 = Area(x_min=np.array([2.25,0.5]),x_max=np.array([2.75,1.0]))

    #     world.dim = 2
    #     world.robots = [robot1,robot2]
    #     world.objects = [object1]

    #     # World bounding box
    #     world.x_lb = np.array([0,-1.75])
    #     world.x_ub = np.array([3.5,1.75])

    #     # Obstacles
    #     world.obstacles= []

    #     # Area's of interest
    #     world.areas = [area1]
    elif specification == "hw_test3":
        tf = 30

        world.spec = Spec(t0=0,tf=tf)
        bz = 5
        
        robot1 = Robot(name="pop",
                       x0=np.array([0.3,0]),
                       dx0=np.array([0,0]),
                       xf=np.array([0.3,0]),
                       dxf= np.array([0,0]),nbz=bz, dq_lb=np.array([-0.2,-0.2]),dq_ub=np.array([0.2,0.2]))
        
        robot2 = Robot(name="crackle",
                      x0=np.array([3.25,0.0]),
                      dx0=np.array([0,0]),
                      xf=None,
                      dxf=np.array([0,0]),nbz=bz, dq_lb=np.array([-0.2,-0.2]),dq_ub=np.array([0.2,0.2]))
        
        object1 = Object(name="snap",
                         x0=np.array([1.0,0.0]),
                         dx0=np.array([0,0]),
                         xf=np.array([2.75,0.0]),
                         dxf=np.array([0,0]),nbz=bz, dq_lb=np.array([-0.2,-0.2]),dq_ub=np.array([0.2,0.2]))
        
        area1 = Area(x_min=np.array([2.25,0.5]),x_max=np.array([2.75,1.0]))

        world.dim = 2
        world.robots = [robot1,robot2]
        world.objects = [object1]

        # World bounding box
        world.x_lb = np.array([0,-1.75])
        world.x_ub = np.array([3.5,1.75])

        # Obstacles
        world.obstacles= []

        # Area's of interest
        world.areas = [area1]