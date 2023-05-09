from random import random

import gym
from gym import error,spaces,utils
from gym.utils import seeding
import numpy as np
import math
from math import *
import random

class muMIMOEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.M = 2
        self.N = 20
        self.K = 2
        self.Ke = 0
        self.EPISODES = 1000
        self.Kfactor = 10
        self.sigma2_BS = .1  # Noise level at BS side
        self.sigma2_UE = .5 # Noise level at UE side

        self.QuantLevel = 8  # Quantization level of Phase shift
        ## Action Set
        ShiftCodebook = [np.exp(1j * pi * 2 * np.arange(0, self.N, 1) / self.N),
                         np.exp(-1j * pi * 2 * np.arange(0, self.N, 1) / self.N),
                         np.exp(3j * pi * 2 * np.arange(0, self.N, 1) / self.N),
                         np.exp(-3j * pi * 2 * np.arange(0, self.N, 1) / self.N),
                         np.exp(0j * pi * 2 * np.arange(0, self.N, 1) / self.N)]
        self.ShiftCodebook = np.array(ShiftCodebook)
        self.state_size = [self.M * self.K * 2, self.N * 2]
        self.action_size = np.size(self.ShiftCodebook)
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Tuple([spaces.Box(low=-1,high=1,shape=(self.M,2*self.K)), spaces.Box(low=-1,high=1,shape=(1,2*self.N))])
        self.state = None
        self.RefVector = np.exp(1j * pi * np.zeros((1, self.N)))   ### self.RefVector 表示当前RIS的状态
        action_init = random.randrange(len(self.ShiftCodebook))
        self.RefVector = self.RefVector * self.ShiftCodebook[action_init, :]

        ## Channel Dynamics
        self.block_duration = 1  ### When block_duration>1, ESC will be applied

        self.Pilot = self.DFT_matrix(self.K)  ## Pilot pattern 导频信号的样式
        self.Reward = None

        self.ArrayShape_BS = [self.M, 1, 1]  ## array shape
        self.ArrayShape_IRS = [1, self.N, 1]  ##
        self.ArrayShape_UE = [1, 1, 1]  ## UE is with 1 antenna
        self.Pos_BS = np.array([0, 0, 10])  # Position of BS
        self.Pos_IRS = np.array([-2, 5, 5])  # Position of IRS
        self.Pos_UE = None    # Position of User and it changes when comes into a new episodes

        print("The Wireless Environment is already established!")

    def _seed(self, seed=None):
        self.np_random, seed = random.seeding.np_random(seed)
        return [seed]

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def reset(self):
        #######################################################################################################
        self.Pos_UE = np.array([[np.random.rand(1) * 10, np.random.rand(1) * 10, 1.5],
                           [np.random.rand(1) * 10, np.random.rand(1) * 10, 1.5]],
                          dtype=np.float)  ## UE positions are randomly generated in each episode
        H_U2B_LoS, H_R2B_LoS, H_U2R_LoS = self.H_GenFunLoS(self.Pos_BS, self.Pos_IRS, self.Pos_UE, self.ArrayShape_BS, self.ArrayShape_IRS,
                                                                 self.ArrayShape_UE)  ## LoS component
        H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS = self.H_GenFunNLoS(self.M, self.N, self.K)
        H_U2B = sqrt(1 / (self.Kfactor + 1)) * H_U2B_NLoS + sqrt(self.Kfactor / (self.Kfactor + 1)) * H_U2B_LoS
        H_R2B = sqrt(1 / (self.Kfactor + 1)) * H_R2B_NLoS + sqrt(self.Kfactor / (self.Kfactor + 1)) * H_R2B_LoS
        H_U2R = sqrt(1 / (self.Kfactor + 1)) * H_U2R_NLoS + sqrt(self.Kfactor / (self.Kfactor + 1)) * H_U2R_LoS
        #######################################################################################################

        H_synt = self.H_syntFun(H_U2B, H_R2B, H_U2R, self.RefVector[0])  ### The aggregated wireless channel
        H_synt_vector = np.reshape(H_synt, (1, self.M * self.K))
        self.state = [np.concatenate((H_synt_vector.real, H_synt_vector.imag), axis=1),
                      np.concatenate((self.RefVector.real, self.RefVector.imag), axis=1)]

        self.Reward, y_rx, H_est = self.GetRewards(self.Pilot, H_synt, self.sigma2_BS, self.sigma2_UE)
        H_est_vector = np.reshape(H_est, (1, self.M * self.K))  # change the est_channel matrix into a vector
        self.Obser = [np.concatenate((H_est_vector.real, H_est_vector.imag), axis=1),
                         np.concatenate((self.RefVector.real, self.RefVector.imag), axis=1)]
        return self.Obser

    def step(self, action):
        ########### update the NOLO of wireless channel
        H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS = self.H_GenFunNLoS(self.M, self.N, self.K)
        H_U2B = sqrt(1 / (self.Kfactor + 1)) * H_U2B_NLoS + sqrt(self.Kfactor / (self.Kfactor + 1)) * self.H_U2B_LoS
        H_R2B = sqrt(1 / (self.Kfactor + 1)) * H_R2B_NLoS + sqrt(self.Kfactor / (self.Kfactor + 1)) * self.H_R2B_LoS
        H_U2R = sqrt(1 / (self.Kfactor + 1)) * H_U2R_NLoS + sqrt(self.Kfactor / (self.Kfactor + 1)) * self.H_U2R_LoS
        ### The aggregated wireless channel
        self.RefVector = self.RefVector * self.ShiftCodebook[action, :]
        ####################################################################################
        H_synt = self.H_syntFun(H_U2B, H_R2B, H_U2R, self.RefVector[0])
        H_synt_vector = np.reshape(H_synt, (1, self.M * self.K))
        self.state = [np.concatenate((H_synt_vector.real, H_synt_vector.imag), axis=1),
                      np.concatenate((self.RefVector.real, self.RefVector.imag), axis=1)]
        ####################################################################################
        self.Reward, y_rx, H_est = self.GetRewards(self.Pilot, H_synt, self.sigma2_BS, self.sigma2_UE)
        H_est_vector = np.reshape(H_est, (1, self.M *self.K))
        self.Obser = [np.concatenate((H_est_vector.real, H_est_vector.imag), axis=1),
                      np.concatenate((self.RefVector.real, self.RefVector.imag), axis=1)]
        ####################################################################################
        return self.Obser, self.Reward

    def render(self, mode='human', close=False):
        print(self.state, self.Obser, self.Reward)


    def DFT_matrix(self, N_point):
        n, m = np.meshgrid(np.arange(N_point), np.arange(N_point))
        omega = np.exp(-2 * pi * 1j / N_point)
        W = np.power(omega, n * m) / sqrt(N_point)  ##
        return W

    def CH_Prop(self, H, sigma2, Pilot):
        NumAnt, NumUser = np.shape(H)
        noise = 1 / sqrt(2) * (np.random.normal(0, sigma2, size=(NumAnt, NumUser)) + 1j * np.random.normal(0, sigma2,
                                                                                      size=(NumAnt, NumUser)))  ## Gaussian Noise
        y_rx = np.dot(H, Pilot) + noise
        return y_rx

    def CH_est(self, y_rx, sigma2, Pilot):
        MMSE_matrix = np.matrix.getH(Pilot) / (1 + sigma2)  ## MMSE channel estimation
        H_est = np.dot(y_rx, MMSE_matrix)
        return H_est

    def Precoding(self, H_est):
        F = np.dot(np.linalg.inv(np.dot(np.matrix.getH(H_est), H_est)),
                   np.matrix.getH(H_est))  ## Zero-forcing Precoding
        NormCoeff = abs(np.diag(np.dot(F, np.matrix.getH(F))))
        NormCoeff = 1 / np.sqrt(NormCoeff)
        F = np.dot(np.diag(NormCoeff), F)  ## Normalization
        return F

    def GetRewards(self, Pilot, H_synt, sigma2_BS, sigma2_UE):
        y_rx = self.CH_Prop(H_synt, sigma2_BS, Pilot)  ### Received singal
        H_est = self.CH_est(y_rx, sigma2_BS, Pilot)  ### Estimated equivalent channel
        F = self.Precoding(H_est)  ### Zero-Forcing precoding
        H_eq = np.dot(F, H_synt)
        H_eq2 = abs(H_eq * np.conj(H_eq))
        SigPower = np.diag(H_eq2)  #### Singal Power
        IntfPower = H_eq2.sum(axis=0)
        IntfPower = IntfPower - SigPower  #### Interference Power
        SINR = SigPower / (IntfPower + sigma2_UE)  ### SNIR
        Rate = np.log2(1 + SINR)  #### Data Rate
        return Rate, y_rx, H_est

    def SubSteeringVec(self, Angle, NumAnt):
        SSV = np.exp(1j * Angle * math.pi * np.arange(0, NumAnt, 1))
        SSV = SSV.reshape(-1, 1)
        return SSV

    def ChannelResponse(self, Pos_A, Pos_B, ArrayShape_A,
                        ArrayShape_B):  ## LoS channel response, which is position dependent
        dis_AB = np.linalg.norm(Pos_A - Pos_B)  ## distance
        DirVec_AB = (Pos_A - Pos_B) / dis_AB  ## direction vector
        angleA = [np.linalg.multi_dot([[1, 0, 0], DirVec_AB]), np.linalg.multi_dot([[0, 1, 0], DirVec_AB]),
                  np.linalg.multi_dot([[0, 0, 1], DirVec_AB])]
        SteeringVectorA = np.kron(self.SubSteeringVec(angleA[0], ArrayShape_A[0]),
                                  self.SubSteeringVec(angleA[1], ArrayShape_A[1]))
        SteeringVectorA = np.kron(SteeringVectorA, self.SubSteeringVec(angleA[2], ArrayShape_A[2]))
        angleB = [np.linalg.multi_dot([[1, 0, 0], DirVec_AB]), np.linalg.multi_dot([[0, 1, 0], DirVec_AB]),
                  np.linalg.multi_dot([[0, 0, 1], DirVec_AB])]
        SteeringVectorB = np.kron(self.SubSteeringVec(angleB[0], ArrayShape_B[0]),
                                  self.SubSteeringVec(angleB[1], ArrayShape_B[1]))
        SteeringVectorB = np.kron(SteeringVectorB, self.SubSteeringVec(angleB[2], ArrayShape_B[2]))
        H_matrix = np.linalg.multi_dot([SteeringVectorA, np.matrix.getH(SteeringVectorB)])
        return H_matrix

    def H_GenFunLoS(self, Pos_BS, Pos_IRS, Pos_UE, ArrayShape_BS, ArrayShape_IRS, ArrayShape_UE):
        NumUE = len(Pos_UE)
        NumAntBS = np.prod(ArrayShape_BS)
        NumEleIRS = np.prod(ArrayShape_IRS)
        H_BU_LoS = np.zeros((NumAntBS, NumUE)) + 1j * np.zeros((NumAntBS, NumUE))
        H_RU_LoS = np.zeros((NumEleIRS, NumUE)) + 1j * np.zeros((NumEleIRS, NumUE))
        for iu in range(NumUE):
            h_BU_LoS = self.ChannelResponse(Pos_BS, Pos_UE[iu], ArrayShape_BS, ArrayShape_UE)
            H_BU_LoS[:, iu] = h_BU_LoS.reshape(-1)
            h_RU_LoS = self.ChannelResponse(Pos_IRS, Pos_UE[iu], ArrayShape_IRS, ArrayShape_UE)
            H_RU_LoS[:, iu] = h_RU_LoS.reshape(-1)
        H_BR_LoS = self.ChannelResponse(Pos_BS, Pos_IRS, ArrayShape_BS, ArrayShape_IRS)
        return H_BU_LoS, H_BR_LoS, H_RU_LoS

    def H_GenFunNLoS(self, NumAntBS, NumEleIRS, NumUser):
        H_U2B_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(NumAntBS, NumUser)) + 1j * np.random.normal(0, 1,
                                                                                    size=(NumAntBS, NumUser)))
        H_R2B_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(NumAntBS, NumEleIRS)) + 1j * np.random.normal(0, 1,
                                                                                    size=(NumAntBS, NumEleIRS)))
        H_U2R_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(NumEleIRS, NumUser)) + 1j * np.random.normal(0, 1,
                                                                                    size=(NumEleIRS, NumUser)))
        return H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS

    def H_syntFun(self, H_U2B, H_R2B, H_U2R, RefVector):  ### Syntheize the aggregated wireless channel
        RefPattern_matrix = np.diag(RefVector)
        H_synt = H_U2B + 1 * np.linalg.multi_dot([H_R2B, RefPattern_matrix, H_U2R])
        return H_synt





