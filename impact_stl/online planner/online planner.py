import casadi as ca
from utilities.beziers import get_derivative_control_points_gurobi, eval_bezier, value_bezier
from utilities.read_write_plan import csv_to_plan, plan_to_csv
# plotting imports
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from matplotlib.patches import Rectangle

import numpy as np
from scipy import optimize

# Set up plot style
plt.style.use(['science'])
rc('text', usetex=True)
rc('font', family='times', size=35)

plt.rcParams['legend.frameon'] = True             # Enable legend frame
plt.rcParams['legend.facecolor'] = 'white'        # Set background color
plt.rcParams['legend.edgecolor'] = 'white'        # Set border color
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc'] = 'best'

class TestOnlinePlanner():
    def __init__(self):
        #Size of world
        self.world_lb = np.array([0,0])
        self.world_ub = np.array([10,10])        
        self.robot_rvars,self.robot_hvars,self.robot_idvars,self.robot_other_names = csv_to_plan(robot_name='snap',
                                                                         scenario_name='minimal_test_throw_and_catch',
                                                                         path='/home/arian/repos/thesis/impact_stl/planner/plans')  

        self.obj_rvars,self.obj_hvars,self.obj_idvars,self.obj_other_names = csv_to_plan(robot_name='pop',
                                                                         scenario_name='minimal_test_throw_and_catch',
                                                                         path='/home/arian/repos/thesis/impact_stl/planner/plans')
        #Size = [nbzs][dim,ncp]
        self.nbzs = len(self.robot_rvars)
        self.dim = 2
        self.world_tf = self.robot_hvars[-1][0,-1]
        #Remkaing to 2d for plotting
        self.obj_rvars = [self.obj_rvars[k][:2,:] for k in range(self.nbzs)]
        self.robot_rvars = [self.robot_rvars[k][:2,:] for k in range(self.nbzs)]
    
        self.robot_drvars = [get_derivative_control_points_gurobi(self.robot_rvars[k], 1) for k in range(self.nbzs)]
        self.obj_drvars = [get_derivative_control_points_gurobi(self.obj_rvars[k], 1) for k in range(self.nbzs)]
        self.robot_dhvars = [get_derivative_control_points_gurobi(self.robot_hvars[k], 1) for k in range(self.nbzs)]
        self.obj_dhvars = [get_derivative_control_points_gurobi(self.obj_hvars[k], 1) for k in range(self.nbzs)]
        
        #self.plot()
        self.compute_trajectories()
        self.animate()

    def compute_trajectories(self):
        N_eval = 100
        self.robot_rtraj = [eval_bezier(self.robot_rvars[k], N_eval) for k in range(self.nbzs)]
        self.obj_rtraj = [eval_bezier(self.obj_rvars[k], N_eval) for k in range(self.nbzs)]
        self.robot_htraj = [eval_bezier(self.robot_hvars[k], N_eval) for k in range(self.nbzs)]
        self.obj_htraj = [eval_bezier(self.obj_hvars[k], N_eval) for k in range(self.nbzs)]
        
        self.robot_drtraj = [eval_bezier(self.robot_drvars[k], N_eval) for k in range(self.nbzs)]
        self.obj_drtraj = [eval_bezier(self.obj_drvars[k], N_eval) for k in range(self.nbzs)]
        self.robot_dhtraj = [eval_bezier(self.robot_dhvars[k], N_eval) for k in range(self.nbzs)]
        self.obj_dhtraj = [eval_bezier(self.obj_dhvars[k], N_eval) for k in range(self.nbzs)]
        self.robot_dqtraj = [np.zeros((self.dim, N_eval)) for k in range(self.nbzs)]
        self.obj_dqtraj = [np.zeros((self.dim, N_eval)) for k in range(self.nbzs)]
        
        for k in range(self.nbzs):
            for i in range(self.dim):
                self.robot_dqtraj[k][i,:] = self.robot_drtraj[k][i,:]/self.robot_dhtraj[k][0,:]
                self.obj_dqtraj[k][i,:] = self.obj_drtraj[k][i,:]/self.obj_dhtraj[k][0,:]


    def plot(self):       
        
        # new layout for results figures
        fig = plt.figure(figsize=(20,10))
        gs = GridSpec(2,4, figure=fig)
        ax1 = fig.add_subplot(gs[0,0:2])
        ax2 = fig.add_subplot(gs[0,2:4])
        ax3 = fig.add_subplot(gs[1,0:2])
        ax4 = fig.add_subplot(gs[1,2:4])
        # ax5 = fig.add_subplot(gs[1,3])

        robot_ls = ['k-','k--','k:','k-.']
        object_ls = ['r-','r--','r:','r-.']
        lw = 4
        s = 15

        # X position vs time
        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:
                ax1.plot(self.obj_htraj[bz][0,:],self.obj_rtraj[bz][0,:],object_ls[0],linewidth=lw)
                ax1.plot(self.obj_htraj[bz][0,0],self.obj_rtraj[bz][0,0],'ro',markersize=s)
        ax1.plot(self.obj_htraj[-1][0,-1],self.obj_rtraj[-1][0,-1],'ro',markersize=s)
        


        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'x' in self.robot_idvars[bz]:
                ax1.plot(self.robot_htraj[bz][0,:],self.robot_rtraj[bz][0,:],'b',linewidth=lw)
                ax1.plot(self.robot_htraj[bz][0,0],self.robot_rtraj[bz][0,0],'bo',markersize=s)
            else:
                ax1.plot(self.robot_htraj[bz][0,:],self.robot_rtraj[bz][0,:],robot_ls[0],linewidth=lw)
                ax1.plot(self.robot_htraj[bz][0,0],self.robot_rtraj[bz][0,0],'ko',markersize=s)
        ax1.plot(self.robot_htraj[-1][0,-1],self.robot_rtraj[-1][0,-1],'ko',markersize=s)
        # ax2.set_title("x-t plane")
        ax1.grid(True)
        ax1.set_xlabel(r"Time [s]")
        ax1.set_ylabel(r"x position [m]")

        # Y position vs time

        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:    
                ax2.plot(self.obj_htraj[bz][0,:],self.obj_rtraj[bz][1,:],object_ls[0],linewidth=lw)
                ax2.plot(self.obj_htraj[bz][0,0],self.obj_rtraj[bz][1,0],'ro',markersize=s)
        ax2.plot(self.obj_htraj[-1][0,-1],self.obj_rtraj[-1][1,-1],'ro',markersize=s)
        
        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'y' in self.robot_idvars[bz] :
                ax2.plot(self.robot_htraj[bz][0,:],self.robot_rtraj[bz][1,:],'b',linewidth=lw)
                ax2.plot(self.robot_htraj[bz][0,0],self.robot_rtraj[bz][1,0],'bo',markersize=s)
            else:
                ax2.plot(self.robot_htraj[bz][0,:],self.robot_rtraj[bz][1,:],robot_ls[0],linewidth=lw)
                ax2.plot(self.robot_htraj[bz][0,0],self.robot_rtraj[bz][1,0],'ko',markersize=s)
        ax2.plot(self.robot_htraj[-1][0,-1],self.robot_rtraj[-1][1,-1],'ko',markersize=s)
        # ax3.set_title("y-t plane")
        ax2.grid(True)
        ax2.set_xlabel(r"Time [s]")
        ax2.set_ylabel(r"y position [m]")

        #X velocity vs time
        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:
                ax3.plot(self.obj_htraj[bz][0,:],self.obj_dqtraj[bz][0,:],object_ls[0],linewidth=lw)
                ax3.plot(self.obj_htraj[bz][0,0],self.obj_dqtraj[bz][0,0],'ro',markersize=s)
        ax3.plot(self.obj_htraj[-1][0,-1],self.obj_dqtraj[-1][0,-1],'ro',markersize=s)
        
        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'x' in self.robot_idvars[bz]:
                ax3.plot(self.robot_htraj[bz][0,:],self.robot_dqtraj[bz][0,:],'b',linewidth=lw)
                ax3.plot(self.robot_htraj[bz][0,0],self.robot_dqtraj[bz][0,0],'bo',markersize=s)
            else:
                ax3.plot(self.robot_htraj[bz][0,:],self.robot_dqtraj[bz][0,:],robot_ls[0],linewidth=lw)
                ax3.plot(self.robot_htraj[bz][0,0],self.robot_dqtraj[bz][0,0],'ko',markersize=s)
        ax3.plot(self.robot_htraj[-1][0,-1],self.robot_dqtraj[-1][0,-1],'ko',markersize=s)
        # ax4.set_title("dx-t plane")
        ax3.grid(True)
        ax3.set_xlabel(r"Time [s]")
        ax3.set_ylabel(r"x velocity [m/s]")

        # Y velocity vs time
        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:
                ax4.plot(self.obj_htraj[bz][0,:],self.obj_dqtraj[bz][1,:],object_ls[0],linewidth=lw)
                ax4.plot(self.obj_htraj[bz][0,0],self.obj_dqtraj[bz][1,0],'ro',markersize=s)
        ax4.plot(self.obj_htraj[-1][0,-1],self.obj_dqtraj[-1][1,-1],'ro',markersize=s)

        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'y' in self.robot_idvars[bz]:
                ax4.plot(self.robot_htraj[bz][0,:],self.robot_dqtraj[bz][1,:],'b',linewidth=lw)
                ax4.plot(self.robot_htraj[bz][0,0],self.robot_dqtraj[bz][1,0],'bo',markersize=s)
            else:
                ax4.plot(self.robot_htraj[bz][0,:],self.robot_dqtraj[bz][1,:],robot_ls[0],linewidth=lw)
                ax4.plot(self.robot_htraj[bz][0,0],self.robot_dqtraj[bz][1,0],'ko',markersize=s)
        ax4.plot(self.robot_htraj[-1][0,-1],self.robot_dqtraj[-1][1,-1],'ko',markersize=s)
        # ax4.set_title("dx-t plane")
        ax4.grid(True)
        ax4.set_xlabel(r"Time [s]")
        ax4.set_ylabel(r"y velocity [m/s]")

        fig.tight_layout()

        #fig.show()

        # x-y plane
        fig_xy = plt.figure(figsize=(10, 10))
        axs = fig_xy.add_subplot(1, 1, 1)
        rect = Rectangle((self.world_lb[0],self.world_lb[1]),self.world_ub[0] - self.world_lb[0],self.world_ub[1] - self.world_lb[1],
                         linewidth=1,edgecolor='k',facecolor='none')
        axs.add_patch(rect)
        for bz in range(self.nbzs):
            axs.plot(self.robot_rtraj[bz][0, :], self.robot_rtraj[bz][1, :], robot_ls[0], linewidth=lw)
            axs.plot(self.robot_rtraj[bz][0, 0], self.robot_rtraj[bz][1, 0], 'ko', markersize=s)
            axs.plot(self.robot_rtraj[-1][0, -1],self.robot_rtraj[-1][1, -1], 'ko', markersize=s)
        for bz in range(self.nbzs):
            axs.plot(self.obj_rtraj[bz][0, :], self.obj_rtraj[bz][1, :], object_ls[0], linewidth=lw)
            axs.plot(self.obj_rtraj[bz][0, 0], self.obj_rtraj[bz][1, 0], 'ro', markersize=s)
            axs.plot(self.obj_rtraj[-1][0, -1], self.obj_rtraj[-1][1, -1], 'ro', markersize=s)
        axs.set_title("x-y plane")
        axs.set_xlabel("x [m]")
        axs.set_ylabel("y [m]")
        axs.grid(True)
        axs.set_aspect('equal', 'box')
        fig_xy.tight_layout()
        
        fig_xy.show()
        plt.pause(100)


    def evaluate_t(self,t):
            t_array_robots = np.array([self.robot_hvars[bzr][0,0] for bzr in range(self.nbzs)])
            t_array_objects = np.array([self.obj_hvars[bzo][0,0] for bzo in range(self.nbzs)]) 

            idxs_robots = np.where(t_array_robots<=t)[0][-1]
            idxs_objects = np.where(t_array_objects<=t)[0][-1]

            errors_robot= lambda s: value_bezier(self.robot_htraj[idxs_robots],s)[0] - t
            errors_object =lambda s: value_bezier(self.obj_htraj[idxs_objects],s)[0] - t

            s_robot = optimize.root_scalar(errors_robot,bracket=[0,1]).root
            s_object = optimize.root_scalar(errors_object,bracket=[0,1]).root

            return idxs_robots, s_robot, idxs_objects, s_object
    
    def animate(self):
        Neval = 250
        self.t_range = np.linspace(0,self.world_tf,Neval)

        self.fig_anim = plt.figure(figsize=(10,10))
        self.ax_anim = plt.axes()
        anim = animation.FuncAnimation(self.fig_anim,self._animate_update,
                                        frames=Neval,interval=self.world_tf/Neval*1e6*2)
        anim.save("/home/arian/repos/thesis/impact_stl/online planner/figures/animation.mp4",writer="ffmpeg", fps=Neval/self.world_tf)


    def _animate_update(self,i):
        t = self.t_range[i]
        idxs_robot, s_robot, idxs_object, s_object = self.evaluate_t(t)

        self.ax_anim.clear()
        rect = Rectangle((self.world_lb[0],self.world_lb[1]),self.world_ub[0] - self.world_lb[0],self.world_ub[1] - self.world_lb[1],
                         linewidth=1,edgecolor='k',facecolor='none')
        self.ax_anim.add_patch(rect)
        # plot circle
        c = (value_bezier(self.robot_rtraj[idxs_robot],s_robot)[0],
            value_bezier(self.robot_rtraj[idxs_robot],s_robot)[1])
        rad = 0.2
        circle = plt.Circle(c,rad,fill=False,color='k')
        self.ax_anim.add_patch(circle)
        self.ax_anim.text(
        c[0], c[1] + 0.25,  # slightly above the circle
        'RobTheRobot',
        color='k', fontsize=12, ha='center', va='bottom', weight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        # plot trajectory 
        for bz in range(self.nbzs):
            #Only plot the traj if interaction
            if 'inter' in self.robot_idvars[bz]:
                self.ax_anim.plot(self.robot_rtraj[bz][0,:],self.robot_rtraj[bz][1,:],'b', linewidth=2)

        # plot circle
        c = (value_bezier(self.obj_rtraj[idxs_object],s_object)[0],
            value_bezier(self.obj_rtraj[idxs_object],s_object)[1])
        rad = 0.3
        circle = plt.Circle(c,rad,fill=False,color='r')
        self.ax_anim.add_patch(circle)
        # plot trajectory
        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:
                self.ax_anim.plot(self.obj_rtraj[bz][0,:],self.obj_rtraj[bz][1,:],'r',linewidth=2)
        self.ax_anim.set_title("x-y plane")
        self.ax_anim.set_xlabel("x [m]")
        self.ax_anim.set_ylabel("y [m]")
        self.ax_anim.set_aspect("equal")
test = TestOnlinePlanner()
