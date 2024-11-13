import cvxpy as cp
import numpy as np
import math
import gurobipy

def Incidence_matrices(num_nodes, num_lines, sending_end, receiving_end):
    
    A_plus = np.zeros((num_lines,num_nodes)) # A+ matrix (num_nodes x num_lines)
    for i in range(num_lines):
        A_plus[i,int(sending_end[i])] = 1
        A_plus[i,int(receiving_end[i])] = -1

    A_minus = np.zeros((num_lines,num_nodes)) # A- matrix (num_nodes x num_lines)
    for i in range(num_lines):
        A_minus[i,int(sending_end[i])] = 0
        A_minus[i,int(receiving_end[i])] = -1

    A_plus = np.transpose(A_plus)
    A_minus = np.transpose(A_minus)

    return A_plus, A_minus




def SOC_ACOPF_single_step(baseMVA, sending_node, receiving_node, R_l, X_l, B_l, p_d, q_d, pn_bound, qn_bound, v_bound, G_n, B_n, K_l, quad_cost, lin_cost, const_cost,
               ESS_soc, ESS_cha_bound, ESS_dis_bound, ESS_soc_bound, 
               theta_n_min=-1, theta_n_max=-1, theta_l_min=-1, theta_l_max=-1, eta_dis = 1, eta_cha = 1):

    ######################################################
    #Function used to compute the SOC-ACOPF (model 2) from the paper XXXXXXX
    """
    This model is very usefull since it allows to compute the optimal power flow for any grid topology and respects the line ampacity of the system
    """
    ######################################################
    """Inputs"""
    # sending_node : array with the sending node with number from 0 to N_bus-1 for line l
    # sending_node : array with the receiving node with number from 0 to N_bus-1 for line l
    # All values in p.u. except the cost functions
    # quad_cost : Quadratic cost coefficient
    # lin_cost : Linear cost coefficient
    # const_cost : Constant cost coefficient
    
    ######################################################
    """Output"""


    ######################################################
    """Initialisation"""
    num_nodes = len(p_d)  # Number of nodes
    num_lines = len(sending_node)  # Number of lines

    A_plus, A_minus = Incidence_matrices(num_nodes,num_lines,sending_node,receiving_node)

    # Unfolding nodes data
    p_n_min = pn_bound[:,0]  # Minimum active power generation at each node
    p_n_max = pn_bound[:,1]  # Maximum active power generation at each node
    q_n_min = qn_bound[:,0]  # Minimum reactive power generation at each node
    q_n_max = qn_bound[:,1]  # Maximum reactive power generation at each node
    V_min = v_bound[:,0]**2  # Minimum voltage squared at each node
    V_max = v_bound[:,1]**2  # Maximum voltage squared at each node
    ESS_cha_min = ESS_cha_bound[:,0] # Minimum charging rate at each node 
    ESS_cha_max = ESS_cha_bound[:,1] # Maximum charging rate at each node
    ESS_dis_min = ESS_dis_bound[:,0] # Minimum discharging rate at each node 
    ESS_dis_max = ESS_dis_bound[:,1] # Maximum discharging rate at each node
    ESS_soc_min = ESS_soc_bound[:,0] # Minimum state of charge at each node 
    ESS_soc_max = ESS_soc_bound[:,1] # Maximum state of charge at each node

    # theta_n min and max
    if theta_n_min == -1 : theta_n_min = -np.pi/2*np.ones(num_nodes)  # Minimum bus angle 
    if theta_n_max == -1 : theta_n_max = np.pi/2*np.ones(num_nodes)  # Maximum bus angle

    # theta_l min and max
    if theta_l_min == -1 : theta_l_min = -np.pi/2*np.ones(num_lines)  # Minimum line angle (from relaxation assumption)
    if theta_l_max == -1 : theta_l_max = np.pi/2*np.ones(num_lines)  # Maximum line angle (from relaxation assumption)

    ######################################################
    """Variables"""

    p_n = cp.Variable(num_nodes)  # Active power at node n
    q_n = cp.Variable(num_nodes)  # Reactive power at node n
    V_n = cp.Variable(num_nodes)  # Voltage magnitude squared at node n
    theta_n = cp.Variable(num_nodes)  # Voltage angles at node n

    p_sl = cp.Variable(num_lines)  # Active power at sending end of line l
    q_sl = cp.Variable(num_lines)  # Reactive power at sending end of line l
    p_ol = cp.Variable(num_lines)  # Active power losses on line l
    q_ol = cp.Variable(num_lines)  # Reactive power losses on line l
    K_ol = cp.Variable(num_lines)  # Branch Equivalent ampacity constraint on line l
    theta_l = cp.Variable(num_lines)  # Voltage angles at line l

    ESS_cha = cp.Variable(num_nodes)
    ESS_dis = cp.Variable(num_nodes)

    #z_l = cp.Variable(num_lines)    # z_l >= sqrt(p_sl^2 + q_sl^2)

    ######################################################
    """ Constraints"""
    constraints = []

    # All constraints located on the buses
    for n in range(num_nodes): 

        # Voltage Magnitude bounds (1k):
        constraints.append(V_n[n] >= V_min[n])
        constraints.append(V_n[n] <= V_max[n])

        # Node angle bounds (1m):
        constraints.append(theta_n[n] >= theta_n_min[n])
        constraints.append(theta_n[n] <= theta_n_max[n])

        # Active power bounds (1n):
        constraints.append(p_n[n] >= p_n_min[n])
        constraints.append(p_n[n] <= p_n_max[n])

        # Reactive power bounds (1o):
        constraints.append(q_n[n] >= q_n_min[n])
        constraints.append(q_n[n] <= q_n_max[n])

        # ESS charging and discharging rate bounds :
        constraints.append(ESS_cha[n] >= ESS_cha_min[n])
        constraints.append(ESS_cha[n] <= ESS_cha_max[n])
        constraints.append(ESS_dis[n] >= ESS_dis_min[n])
        constraints.append(ESS_dis[n] <= ESS_dis_max[n])

        # ESS min and max SOC
        constraints.append(ESS_soc[n] + ESS_cha[n] <= ESS_soc_max[n])
        constraints.append(ESS_soc[n] - ESS_dis[n] >= ESS_soc_min[n])

        # Active power balance (1b):
        #constraints.append(p_n[n] - p_d[n] == A_plus[n, :]@p_sl - A_minus[n, :]@p_ol + G_n[n]*V_n[n])
        constraints.append(p_n[n] + ESS_dis[n] - p_d[n] - ESS_cha[n] == A_plus[n, :]@p_sl - A_minus[n, :]@p_ol + G_n[n]*V_n[n])

        # Reactive power balance (1c):
        constraints.append(q_n[n] - q_d[n] == A_plus[n, :]@q_sl - A_minus[n, :]@q_ol - B_n[n]*V_n[n])




    #All contrsiants located on the lines
    for l in range(num_lines): 

        # Line angle bounds (1l):
        constraints.append(theta_l[l] >= theta_l_min[l])
        constraints.append(theta_l[l] <= theta_l_max[l])

        # Voltage drop constraint (1d):
        constraints.append(V_n[sending_node[l]] - V_n[receiving_node[l]] == 2*R_l[l]*p_sl[l] + 2*X_l[l]*q_sl[l] - R_l[l]*p_ol[l] - X_l[l]*q_ol[l])

        # Conic active and reactive power losses constraint (2b):
        constraints.append(K_ol[l] == (K_l[l] - V_n[sending_node[l]]*B_l[l]**2 + 2*q_sl[l]*B_l[l])*X_l[l])
        constraints.append(K_ol[l] >= q_ol[l])
        #constraints.append(cp.SOC(z_l[l], cp.hstack([p_sl[l], q_sl[l]])))
        #constraints.append(cp.norm(cp.vstack([2*z_l[l], q_ol[l] - V_n[sending_node[l]]/X_l[l]]),2) <= q_ol[l] + V_n[sending_node[l]]/X_l[l])
        constraints.append(
            cp.norm(
                cp.vstack([
                    2 * np.sqrt(X_l[l]) * cp.vstack([p_sl[l], q_sl[l]]),
                    cp.reshape(q_ol[l] - V_n[sending_node[l]], (1, 1))
                ]),
                2
            ) <= q_ol[l] + V_n[sending_node[l]]
        )

        # Power loss constraint (2c):
        constraints.append(p_ol[l] * X_l[l] == q_ol[l] * R_l[l])

        # Line angle constraint (1h):
        constraints.append(theta_l[l] == theta_n[sending_node[l]] - theta_n[receiving_node[l]])

        # Linearized angle constraint (2d):
        constraints.append(theta_l[l] == X_l[l]*p_sl[l] - R_l[l]*q_sl[l])

        # Feasibility solution recovery equation (4g):
        constraints.append(V_n[sending_node[l]] + V_n[receiving_node[l]] >= cp.norm(cp.vstack([2*theta_l[l]/np.sin(theta_l_max[l]), V_n[sending_node[l]] - V_n[receiving_node[l]]]),2))

        #Try to constraint each line power and reactive power losses:
        #constraints.append(p_sl[l] >= -sum(p_d))
        #constraints.append(p_sl[l] <= sum(p_d))
        #constraints.append(q_sl[l] >= -sum(p_d))
        #constraints.append(q_sl[l] <= sum(p_d))


    #####################################################################
    """Objective Function""" 
    # Minimize total generation cost by using quadratic relationship

    objective = 0
    for i in range(num_nodes):
        objective += quad_cost[i]*cp.square(p_n[i]*baseMVA) + lin_cost[i]*p_n[i]*baseMVA + const_cost[i]

    # Defining the optimization problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    #####################################################################

    #####################################################################
    # Solve the problem
    problem.solve(
        solver=cp.SCS,
        #NumericFocus=3,
        #QCPDual=1,
        #TimeLimit=600,        # Allows up to 10 minutes
        #MIPGap=0.01,          # Sets a relative MIP gap of 1%
        #OptimalityTol=1e-5, 
        #verbose=True
    )
    print("Problem Status:", problem.status)
    print("Optimal Value of the Objective Function:", problem.value)

    return problem.value, p_n.value, q_n.value, np.sqrt(V_n.value), p_sl.value, q_sl.value, p_ol.value, q_ol.value, K_ol.value, theta_n.value, theta_l.value, ESS_cha.value, ESS_dis.value
