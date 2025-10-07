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
import numpy as np
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
        
        
        self.robot_rvars,self.robot_hvars,self.robot_idvars,self.robot_other_names = csv_to_plan(robot_name='snap',
                                                                         scenario_name='minimal_test_throw_and_catch',
                                                                         path='/home/arian/repos/thesis/impact_stl/planner/plans')  

        self.obj_rvars,self.obj_hvars,self.obj_idvars,self.obj_other_names = csv_to_plan(robot_name='pop',
                                                                         scenario_name='minimal_test_throw_and_catch',
                                                                         path='/home/arian/repos/thesis/impact_stl/planner/plans')
        #Size = [nbzs][dim,ncp]
        self.nbzs = len(self.robot_rvars)
        self.dim = 2

        #Remkaing to 2d for plotting
        self.obj_rvars = [self.obj_rvars[k][:2,:] for k in range(self.nbzs)]
        self.robot_rvars = [self.robot_rvars[k][:2,:] for k in range(self.nbzs)]
    
        self.robot_drvars = [get_derivative_control_points_gurobi(self.robot_rvars[k], 1) for k in range(self.nbzs)]
        self.obj_drvars = [get_derivative_control_points_gurobi(self.obj_rvars[k], 1) for k in range(self.nbzs)]
        self.robot_dhvars = [get_derivative_control_points_gurobi(self.robot_hvars[k], 1) for k in range(self.nbzs)]
        self.obj_dhvars = [get_derivative_control_points_gurobi(self.obj_hvars[k], 1) for k in range(self.nbzs)]
        
        self.plot()
        



    def plot(self):
        print('hello')
        N_eval = 100
        robot_rtraj = [eval_bezier(self.robot_rvars[k], N_eval) for k in range(self.nbzs)]
        obj_rtraj = [eval_bezier(self.obj_rvars[k], N_eval) for k in range(self.nbzs)]
        robot_htraj = [eval_bezier(self.robot_hvars[k], N_eval) for k in range(self.nbzs)]
        obj_htraj = [eval_bezier(self.obj_hvars[k], N_eval) for k in range(self.nbzs)]
        
        robot_drtraj = [eval_bezier(self.robot_drvars[k], N_eval) for k in range(self.nbzs)]
        obj_drtraj = [eval_bezier(self.obj_drvars[k], N_eval) for k in range(self.nbzs)]
        robot_dhtraj = [eval_bezier(self.robot_dhvars[k], N_eval) for k in range(self.nbzs)]
        obj_dhtraj = [eval_bezier(self.obj_dhvars[k], N_eval) for k in range(self.nbzs)]
        robot_dqtraj = [np.zeros((self.dim, N_eval)) for k in range(self.nbzs)]
        obj_dqtraj = [np.zeros((self.dim, N_eval)) for k in range(self.nbzs)]
        
        for k in range(self.nbzs):
            for i in range(self.dim):
                robot_dqtraj[k][i,:] = robot_drtraj[k][i,:]/robot_dhtraj[k][0,:]
                obj_dqtraj[k][i,:] = obj_drtraj[k][i,:]/obj_dhtraj[k][0,:]
        
        
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
                ax1.plot(obj_htraj[bz][0,:],obj_rtraj[bz][0,:],object_ls[0],linewidth=lw)
                ax1.plot(obj_htraj[bz][0,0],obj_rtraj[bz][0,0],'ro',markersize=s)
        ax1.plot(obj_htraj[-1][0,-1],obj_rtraj[-1][0,-1],'ro',markersize=s)
        


        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'x' in self.robot_idvars[bz]:
                ax1.plot(robot_htraj[bz][0,:],robot_rtraj[bz][0,:],'b',linewidth=lw)
                ax1.plot(robot_htraj[bz][0,0],robot_rtraj[bz][0,0],'bo',markersize=s)
            else:
                ax1.plot(robot_htraj[bz][0,:],robot_rtraj[bz][0,:],robot_ls[0],linewidth=lw)
                ax1.plot(robot_htraj[bz][0,0],robot_rtraj[bz][0,0],'ko',markersize=s)
        ax1.plot(robot_htraj[-1][0,-1],robot_rtraj[-1][0,-1],'ko',markersize=s)
        # ax2.set_title("x-t plane")
        ax1.grid(True)
        ax1.set_xlabel(r"Time [s]")
        ax1.set_ylabel(r"x position [m]")

        # Y position vs time

        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:    
                ax2.plot(obj_htraj[bz][0,:],obj_rtraj[bz][1,:],object_ls[0],linewidth=lw)
                ax2.plot(obj_htraj[bz][0,0],obj_rtraj[bz][1,0],'ro',markersize=s)
        ax2.plot(obj_htraj[-1][0,-1],obj_rtraj[-1][1,-1],'ro',markersize=s)
        
        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'y' in self.robot_idvars[bz] :
                ax2.plot(robot_htraj[bz][0,:],robot_rtraj[bz][1,:],'b',linewidth=lw)
                ax2.plot(robot_htraj[bz][0,0],robot_rtraj[bz][1,0],'bo',markersize=s)
            else:
                ax2.plot(robot_htraj[bz][0,:],robot_rtraj[bz][1,:],robot_ls[0],linewidth=lw)
                ax2.plot(robot_htraj[bz][0,0],robot_rtraj[bz][1,0],'ko',markersize=s)
        ax2.plot(robot_htraj[-1][0,-1],robot_rtraj[-1][1,-1],'ko',markersize=s)
        # ax3.set_title("y-t plane")
        ax2.grid(True)
        ax2.set_xlabel(r"Time [s]")
        ax2.set_ylabel(r"y position [m]")

        #X velocity vs time
        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:
                ax3.plot(obj_htraj[bz][0,:],obj_dqtraj[bz][0,:],object_ls[0],linewidth=lw)
                ax3.plot(obj_htraj[bz][0,0],obj_dqtraj[bz][0,0],'ro',markersize=s)
        ax3.plot(obj_htraj[-1][0,-1],obj_dqtraj[-1][0,-1],'ro',markersize=s)
        
        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'x' in self.robot_idvars[bz]:
                ax3.plot(robot_htraj[bz][0,:],robot_dqtraj[bz][0,:],'b',linewidth=lw)
                ax3.plot(robot_htraj[bz][0,0],robot_dqtraj[bz][0,0],'bo',markersize=s)
            else:
                ax3.plot(robot_htraj[bz][0,:],robot_dqtraj[bz][0,:],robot_ls[0],linewidth=lw)
                ax3.plot(robot_htraj[bz][0,0],robot_dqtraj[bz][0,0],'ko',markersize=s)
        ax3.plot(robot_htraj[-1][0,-1],robot_dqtraj[-1][0,-1],'ko',markersize=s)
        # ax4.set_title("dx-t plane")
        ax3.grid(True)
        ax3.set_xlabel(r"Time [s]")
        ax3.set_ylabel(r"x velocity [m/s]")

        # Y velocity vs time
        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:
                ax4.plot(obj_htraj[bz][0,:],obj_dqtraj[bz][1,:],object_ls[0],linewidth=lw)
                ax4.plot(obj_htraj[bz][0,0],obj_dqtraj[bz][1,0],'ro',markersize=s)
        ax4.plot(obj_htraj[-1][0,-1],obj_dqtraj[-1][1,-1],'ro',markersize=s)

        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'y' in self.robot_idvars[bz]:
                ax4.plot(robot_htraj[bz][0,:],robot_dqtraj[bz][1,:],'b',linewidth=lw)
                ax4.plot(robot_htraj[bz][0,0],robot_dqtraj[bz][1,0],'bo',markersize=s)
            else:
                ax4.plot(robot_htraj[bz][0,:],robot_dqtraj[bz][1,:],robot_ls[0],linewidth=lw)
                ax4.plot(robot_htraj[bz][0,0],robot_dqtraj[bz][1,0],'ko',markersize=s)
        ax4.plot(robot_htraj[-1][0,-1],robot_dqtraj[-1][1,-1],'ko',markersize=s)
        # ax4.set_title("dx-t plane")
        ax4.grid(True)
        ax4.set_xlabel(r"Time [s]")
        ax4.set_ylabel(r"y velocity [m/s]")

        fig.tight_layout()

        #fig.show()

        # x-y plane
        from matplotlib.patches import Rectangle

        fig_xy = plt.figure(figsize=(10, 10))
        axs = fig_xy.add_subplot(1, 1, 1)
        rect = Rectangle((0,0),10-0,10-0,
                         linewidth=1,edgecolor='k',facecolor='none')
        axs.add_patch(rect)
        for bz in range(self.nbzs):
            axs.plot(robot_rtraj[bz][0, :], robot_rtraj[bz][1, :], robot_ls[0], linewidth=lw)
            axs.plot(robot_rtraj[bz][0, 0], robot_rtraj[bz][1, 0], 'ko', markersize=s)
            axs.plot(robot_rtraj[-1][0, -1],robot_rtraj[-1][1, -1], 'ko', markersize=s)
        for bz in range(self.nbzs):
            axs.plot(obj_rtraj[bz][0, :], obj_rtraj[bz][1, :], object_ls[0], linewidth=lw)
            axs.plot(obj_rtraj[bz][0, 0], obj_rtraj[bz][1, 0], 'ro', markersize=s)
            axs.plot(obj_rtraj[-1][0, -1], obj_rtraj[-1][1, -1], 'ro', markersize=s)
        axs.set_title("x-y plane")
        axs.set_xlabel("x [m]")
        axs.set_ylabel("y [m]")
        axs.grid(True)
        axs.set_aspect('equal', 'box')
        fig_xy.tight_layout()
        
        fig_xy.show()
        plt.pause(100)
test = TestOnlinePlanner()
