#ifndef MPM2D_H
#define MPM2D_H
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <array>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/SparseLU>

using namespace std;
using namespace Eigen;

class mpm2D
{
public:
	/// Default Constructor:
	mpm2D();
	/// Constructor
	mpm2D(string& a_fileParticles);
	///
	void initializeFEMgrid(string& a_fileNodes, string& a_fileConn, string& a_fileNodesP);
	///
	void initializeMass(const double& a_rho_init, double& a_TOL_ratio);
	///
    void initializeVelocity(const double& a_v0);
    ///
	void initializeTemperature(const double& a_T0);
	///
	void initializeRESIDUAL();
	///
	void particleShift(double& a_x, double& a_y);
    ///
    void particleToNodes(const double& a_T_sol, const double& a_T_liq, const double& a_c_sol, const double& a_c_liq, const double& a_LH, const double& a_k_sol, const double& a_k_liq, const double& a_hp, const double& a_hwall, const double& a_Tair, const double& a_Twall, vector<int> a_bCells);
    ///
	void NodalTemperatureAndDensity(const double& a_deltaT, const double& a_rho_ref, const double& a_alpha, const double& a_T_ref);
	///
    void intNodalVelocity(const double& a_deltaT);
	///
	void applyNoSlipBC(VectorXi& a_bcZero);
	///
	void updateParticleTempAndDensity(double& a_deltaT, const double& rho_ref, const double a_T_ref,const double& a_alpha);
	///
	void updateParticleTempAndDensity(const double& a_deltaT, const double& rho_ref, const double& a_T_ref,const double& a_alpha);
	///
	void updateNodalVelocity(MatrixXd& a_gradP, const double& a_deltaT, VectorXi& a_bcZero);
	///
	void nodesToParticles(const double& a_deltaT, const double& a_T_sol, const double& a_T_liq, const double& a_mu_sol, const double& a_mu_liq, const double& a_k_sol, const double& a_k_liq);
	///
	void addParticles(string& a_fileParticles, const double& a_rho_init, const double& a_T0);
	///
	void checkVelocity(const double& a_deltaT, const double& a_nozzle_level, const double& a_v0,  const double& a_x_max, const double& a_v_nozzle_max, const double& a_T_init);
	///
	void writeResults(string& a_frameFileName);
	///
	void writeResultsRESIDUAL(string& a_frameFileName);
	///
	void updateNodalVelAndTempRESIDUAL(const double& a_deltaT, VectorXi& a_bcZero);
	///
	void updateParticleTempAndDensityRESIDUAL(const double& a_deltaT, const double& a_rho_ref, const double& a_T_ref, const double& a_alpha);
	///
	void nodesToParticlesRESIDUAL(const double& a_deltaT, const double& a_T_sol, const double& a_T_liq, const double& a_T_ref, const double& a_k_sol, const double& a_k_liq, double& a_sigma_y, double& a_H, double& a_mu, double& a_K, const double& a_beta);
	///
	inline int get_np(){return np;};
	inline int get_nel(){return nel;};
	inline int get_nn(){return nn;};
	inline int get_el1(){return el1;};
	inline int get_el2(){return el2;};
	inline double get_h1(){return h1;};
	inline double get_h2(){return h2;};
	inline double** get_rp(){return rp;};
	inline double** get_vp(){return vp;};
	inline double* get_Vp(){return Vp;};
	inline double* get_Vp0(){return Vp0;};
	inline double* get_Mp(){return Mp;};
	inline double** get_Fp(){return Fp;};
	inline double** get_Sp(){return Sp;};
    inline double* get_rhop(){return rhop;};
	inline double* get_Tp(){return Tp;};
	inline double** get_qp(){return qp;};
	inline MatrixXd get_vn(){return vn;};
	inline MatrixXd get_v1(){return v1;};
	inline VectorXd get_Tn(){return Tn;};
	inline VectorXi get_edge1(){return e1;};
	inline VectorXi get_edge2(){return e2;};
	inline VectorXi get_edge3(){return e3;};
	inline VectorXi get_edge4(){return e4;};
	inline Vector4i get_corners(){return c;};
	inline VectorXi get_inner(){return inner;};
	inline MatrixXd get_vStar(){return n_vStar;};
	inline MatrixXd get_nodes(){return nodes;};
	inline MatrixXd get_nodesP(){return nodesP;};
	inline MatrixXi get_conn(){return conn;};
	inline VectorXi get_n1(){return n1;};
	inline VectorXi get_n2(){return n2;};
	inline VectorXi get_n3(){return n3;};
	inline VectorXi get_n4(){return n4;};
	inline Vector4i get_nc(){return nc;};
	inline VectorXi get_ninner(){return ninner;};
	inline double get_TOL(){return TOL;};
	inline int get_ADD(){return addMaterial;};
	inline double get_xNozzle(){return x_nozzle;};
	
private:
	///////////////////
	///PARTICLE DATA///
	///////////////////
	int np;         ///Number of total particles
	double TOL;     ///Tolerance for NodesToParticles mapping
	double **rp;    ///Particle Position
	double **vp;    ///Particle Velocity
	double *Vp;     ///Current Particle Volume
	double *Vp0;    ///Reference Particle Volume
	double *Mp;     ///Particle Mass
	double **Fp;    ///Particle Deformation Gradient Tensor
	double **Sp;    ///Particle Stress Tensor
	double *Svmp;   ///Particle Von-Mises Stress
	double **Ep;    ///Particle Infinitesimal Strain Tensor
	double **Ep_pl; ///Particle Plastic Strain Tensor
    double *rhop;   ///Particle Density
	double *Tp;     ///Particle Temperature at the current time step
	double *Tp1;    ///Particle Temperature at the next time step
	double *alphap; ///Particle Plasticity Parameter
	double **qp;    ///Particle Heat Flux
	double **Np;    ///Particle Shape FUnctions
	double **dNxp;  ///Partcile Gradient of Shape Functions - x
	double **dNyp;  ///Particle Gradient of Shape Functions - y
	int *elNp;      ///Particle-Element Connectivity

    //////////////////
	///NODAL VALUES///
	//////////////////
	VectorXd n_mass;     ///Nodal Mass
	MatrixXd n_mom;      ///Nodal Momentum
	MatrixXd n_fi;       ///Nodal Internal Force
	MatrixXd n_fe;       ///Nodal External Force
    VectorXd n_c;        ///Nodal Energy w/out Temperature
    VectorXd n_cT;	     ///Nodal Energy w/ Temperature
    VectorXd n_qi;       ///Nodal Internal Heat
    VectorXd n_qe;       ///Nodal External Heat
	MatrixXd n_momStar;  ///Nodal Intermediate Momentum
	MatrixXd n_vStar;    ///Nodal Intermediate Velocity
    VectorXd n_Energy;   ///Nodal Energy at the next time step
	MatrixXd vn;	     ///Nodal Velocity at the current time step
	MatrixXd v1;		 ///Nodal Velocity at the next time step
	MatrixXd an;		 ///Nodal Acceleration
	VectorXd Tn;         ///Nodal Temperature
	VectorXd rhon;       ///Nodal Density

	/////////////////
	////MESH DATA////
	/////////////////
	MatrixXd nodes;
	MatrixXd nodesP;
	MatrixXi conn;
    int el1;
    int el2;
    double h1;
    double h2;
    int nn;
    int nel;
    int dof;
    ///Elements:
	VectorXi e1;
	VectorXi e2;
	VectorXi e3;
	VectorXi e4;
	Vector4i c;
	VectorXi inner;
	///Nodes:
	VectorXi n1;
	VectorXi n2;
	VectorXi n3;
	VectorXi n4;
	Vector4i nc;
	VectorXi ninner;
	///////////////////////////
	////CHECKING PARAMETERS////
	///////////////////////////
	int addMaterial;
	int deleteParticlesEnd;
	double x_nozzle;         ///Nozzle Position
	double v_nozzle;         ///Nozzle Velocity
};
/////////////////////
///SHAPE FUNCTIONS///
/////////////////////
inline Vector4d get2DlinearShape(double& xi, double& eta){
	Vector4d N;
	N(0) = 0.25*(1.0-xi)*(1.0-eta);
	N(1) = 0.25*(1.0+xi)*(1.0-eta);
	N(2) = 0.25*(1.0+xi)*(1.0+eta);
	N(3) = 0.25*(1.0-xi)*(1.0+eta);
	return N;
}
inline MatrixXd getGrad2DlinearShape(double& xi, double& eta){
	MatrixXd dN(2,4);
	dN(0,0) = -0.25*(1.0-eta);
	dN(0,1) =  0.25*(1.0-eta);
	dN(0,2) =  0.25*(1.0+eta);
	dN(0,3) = -0.25*(1.0+eta);
	dN(1,0) = -0.25*(1.0-xi);
	dN(1,1) = -0.25*(1.0+xi);
	dN(1,2) =  0.25*(1.0+xi);
	dN(1,3) =  0.25*(1.0-xi);
	return dN;		
}
///////////////////////////
inline int find(VectorXi& a_edge, int& cNo){
	int f = 0;
	for (int i = 0; i < a_edge.size(); i++) {
		if(a_edge(i)==cNo){
			f = 1;
			break;
		};
	};
	return f;
};
////////////////////////////
inline int find2(vector<int>& a_edge, int& cNo){
	int f = 0;
	for (int i = 0; i < a_edge.size(); i++) {
		if(a_edge[i]==cNo){
			f = 1;
			break;
		};
	};
	return f;
};
inline int findIndex(vector<int>& a_inner, int& pNo){
	int index;
	for (int i = 0; i < a_inner.size(); i++) {
		if (a_inner[i]==pNo) {
			index = i;
		};
	};
	return index;
};
///////////////////////////
//PHASE CHANGE PARAMETERS//
///////////////////////////
inline double apparentHeatCapacity(double& a_T, const double a_T_sol, const double a_T_liq, const double& a_c_sol, const double& a_c_liq, const double& a_LH){
	double cAPP;
	if(a_T>=a_T_liq)
	{
		cAPP = a_c_liq;
	}else if ( (a_T<a_T_liq) && (a_T>a_T_sol) )
	{
		cAPP = a_c_sol+(a_LH/(a_T_liq-a_T_sol))+(a_c_liq-a_c_sol)*((a_T-a_T_sol)/(a_T_liq-a_T_sol));
	}else
	{
		cAPP = a_c_liq;
	}
	return cAPP;
}
inline double viscosity(double& a_T, const double& a_T_sol, const double& a_T_liq, const double& a_mu_sol, const double& a_mu_liq){
	double lambda;
	double mu;
	if (a_T >= a_T_liq)
	{
		lambda = 1.0;
	}else if ( (a_T<a_T_liq) && (a_T>a_T_sol) )
	{
		lambda = 1.0-(a_T_liq-a_T)/(a_T_liq-a_T_sol);
	}else
	{
		lambda = 0.0;
	}
	mu = a_mu_sol*(1-lambda)+a_mu_liq*lambda;
	return mu;
};
inline double thermalConductivity(double& a_T, const double& a_T_sol, const double& a_T_liq, const double& a_k_sol, const double& a_k_liq){
	double lambda;
	double k;
	if(a_T>=a_T_liq)
	{
		lambda = 1.0;
	}else if ( (a_T<a_T_liq) && (a_T>a_T_sol) )
	{
		lambda = 1-((a_T_liq-a_T)/(a_T_liq-a_T_sol));
	}else
	{
		lambda = 0.0;
	};
	k = a_k_sol*(1-lambda)+a_k_liq*lambda;
	return k;
};
#endif
