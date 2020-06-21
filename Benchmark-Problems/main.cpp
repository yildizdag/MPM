//  Created by Erden Yildizdag on 7/14/18.
//  Copyright © 2018 Erden Yildizdag. All rights reserved.

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <array>
#include <cstring>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include "ppe2D.H"
#include "mpm2D.H"
#include "levelSet.H"

using namespace std;
using namespace Eigen;


int main(int argc, const char* argv[]){

	string fileNameToWrite = "MPM_FDM_THESIS_1_";

	//////////////////////
	//PROCESS PARAMETERS//
	//////////////////////

	///AVERAGE VELOCITY:
	const double vAVE = 0.1;
	///EXTRUSION VELOCITY:
	const double v0 = 2*vAVE;
	///NOZZLE VELOCITY:
	const double v_nozzle = 0.2;
	///DISTANCE TO STOP EXTRUSION:
	const double x_max = 0.0025;
	///NOZZLE LEVEL:
	const double nozzle_level = 0.0004;
	///Tolerance Ratio:
	double TOL_ratio = 0.02;

	///////////////////////
	//MATERIAL PARAMETERS//
	///////////////////////
	const double mu_sol = 0.5;
	const double mu_liq = 0.02;
	const double h_air = 200.0;
	const double h_wall = 2000.0;
	const double gamma = 0.0;
	const double beta = 40E-6;
	const double T_init = 493.0;
	const double T_ref = 293.0;
	const double T_sol = 353.0;
	const double T_liq = 493.0;
	const double T_air = 293.0;
	const double T_wall = 323.0;
	const double rho_ref = 1000;
	const double rho_init = rho_ref*(1-beta*(T_init-T_ref));
	const double c_sol = 1200.0;
	const double c_liq = 1800.0;
	const double LH = 0;
	const double k_sol = 0.2;
	const double k_liq = 0.2;
	double E = 3.5E9;
	double Et = 2E9;
	double nu = 0.35;
	double mu = E/(2*(1+nu));
	double K = E/(3*(1-2*nu));
	double sigma_y = 50E6;
	double H = Et/(1-Et/E);
    //////////////////////////////
	//SET UP THE BACKGROUND GRID//
	//////////////////////////////
	string fileNodes = "nodes.txt";
	string fileConn = "conn.txt";
	string fileNodesP = "nodesP.txt";
	//////////////////////////////////////////////////
	//READ PARTICLES and INITIALIZE PARTICLE STORAGE//
	//////////////////////////////////////////////////
	string fileParticles = "mpm_FDM_THESIS.txt";
	///MPM DATA:
	mpm2D mpmData(fileParticles);
	///FDM DATA:
	ppe2D presData;
	///INITIALIZE FEM GRID:
	mpmData.initializeFEMgrid(fileNodes,fileConn,fileNodesP);
	//SHIFT (if necessary):
	double x_shift = 0.0003;
	double y_shift = nozzle_level+0.0001;
	mpmData.particleShift(x_shift,y_shift);
	//Boundary Conditions:
	VectorXi bcZero = mpmData.get_n1();
	//Velocity:
	mpmData.initializeVelocity(v0);
	//Mass, Density, Deformation Gradient, Stress:
	mpmData.initializeMass(rho_init,TOL_ratio);
	//Temperature and Heat Flux:
	mpmData.initializeTemperature(T_init);
	////////////////////////
	//INITIALIZE LEVEL SET//
	////////////////////////
	levelSet lSet;
	lSet.initializeLevelSet(mpmData);
	////////////////////////////////////////
	////////////////////////////////////////
	////////////// MAIN MPM ////////////////
	////////////////////////////////////////
	////////////////////////////////////////
	const double deltaT = 0.0000005;
	double t = 0.0;
	double time = 0.05;
	double frame = 100;
	double Tstep = time/frame;
	int step = int(floor(Tstep/deltaT));
	int count = 0;
	int wCount = 0;
	int reInitStep = 25;
	int deleteParticleStep = 2000;
	int startResidual = 0;
	double x_current_pos;
	////////////////////////////////////////////
	/////////////// EXTRUSION //////////////////
	////////////////////////////////////////////
	while (t<=time){
		if (count%1000 == 0)
		{
		cout << t << "\n";
		};
		if(count%step == 0){
			string index = to_string(wCount);
			string frameFileName = fileNameToWrite+index+".txt";
			mpmData.writeResults(frameFileName);
			wCount += 1;
		};
		count += 1;
		//////////////////////////
		lSet.detectGhostCells(mpmData);
		vector<int> bCells = lSet.get_bCells();
		//////////////////////////
		////PARTICLES TO NODES////
		//////////////////////////
		mpmData.particleToNodes(T_sol,T_liq,c_sol,c_liq,LH,k_sol,k_liq,h_air,h_wall,T_air,T_wall,bCells);
		///////////////////////////////
		//INTERMEDIATE NODAL VELOCITY//
		///////////////////////////////
		mpmData.intNodalVelocity(deltaT);
		/////////////////////////////////
		//UPDATE NODAL TEMP AND DENSITY//
		/////////////////////////////////
		mpmData.NodalTemperatureAndDensity(deltaT,rho_ref,beta,T_ref);
		/////////////
		//APPLY BCs//
		/////////////
		mpmData.applyNoSlipBC(bcZero);
		///////////////////////////////////////////
		//UPDATE PARTICLE TEMPERATURE AND DENSITY//
		///////////////////////////////////////////
		mpmData.updateParticleTempAndDensity(deltaT,rho_ref,T_ref,beta);
		///////////////////////////
		//PRESSURE POISSON SOLVER//
		///////////////////////////
		presData.ppe2Dsolver(mpmData,lSet,deltaT,rho_ref,beta,T_ref,gamma);
		presData.presGrad2DAtNodes(mpmData);
		/////////////////////////
		//UPDATE VELOCITY FIELD//
		/////////////////////////
		MatrixXd gradP = presData.get_gradP();
		mpmData.updateNodalVelocity(gradP,deltaT,bcZero);
		////////////////////
		//LEVEL SET UPDATE//
		////////////////////
		lSet.levelSetUpdateUpwind(mpmData,deltaT);
		//////////////////////
		//NODES TO PARTICLES//
		//////////////////////
		mpmData.nodesToParticles(deltaT,T_sol,T_liq,mu_sol,mu_liq,k_sol,k_liq);
		//////////////////////
		/////REINITIALIZE/////
		//////////////////////
		if (count%reInitStep == 0){
			lSet.initializeLevelSet(mpmData);
		};
		//////////////////////
		///DELETE and ADD/////
		//////////////////////
		int addMat = mpmData.get_ADD();
		if (count%deleteParticleStep == 0 && addMat == 1){
			mpmData.addParticles(fileParticles,rho_init,T_init);
		};
		/////////////////////
		///CHECK VELOCITY////
		/////////////////////
		mpmData.checkVelocity(deltaT,nozzle_level,v0,x_max,v_nozzle,T_init);
		///UPDATE TIME:
	   	t += deltaT;
		///CHECK FOR RESIDUAL:
		x_current_pos = mpmData.get_xNozzle();
		if ((x_current_pos>x_max) && (startResidual==0))
		{
			time = t + 0.001;
			startResidual = 1;
		};
	};
	cout << "EXTRUSION ENDS"  << "\n";
	///Final Time for Residual Analysis:
	time += 0.05;
	///Initialize Residual Analysis:
	mpmData.initializeRESIDUAL(); 
	/////////////////////////////////////////////////
	/////////////// RESIDUAL ANALYSIS ///////////////
	/////////////////////////////////////////////////
	cout << "RESIDUAL STARTS"  << "\n";
	wCount = 0;
	const double deltaTT = 2E-8;
	while (t<=time)
	{
		if (count%1000 == 0)
		{
		cout << t << "\n";
		};
		if(count%step == 0){
			string index = to_string(wCount);
			string frameFileName = fileNameToWrite+"RESIDUAL_"+index+".txt";
			mpmData.writeResultsRESIDUAL(frameFileName);
			wCount += 1;
		};
		count += 1;
		//////////////////////////
		lSet.detectGhostCells(mpmData);
		vector<int> bCells = lSet.get_bCells();
		//////////////////////////
		////PARTICLES TO NODES////
		//////////////////////////
		mpmData.particleToNodes(T_sol,T_liq,c_sol,c_liq,LH,k_sol,k_liq,h_air,h_wall,T_air,T_wall,bCells);
		//////////////////////////////////
		//NODAL VELOCITY and TEMPERATURE//
		//////////////////////////////////
		mpmData.updateNodalVelAndTempRESIDUAL(deltaTT,bcZero);
		///////////////////////////////////////////
		//UPDATE PARTICLE TEMPERATURE AND DENSITY//
		///////////////////////////////////////////
		mpmData.updateParticleTempAndDensityRESIDUAL(deltaTT,rho_ref,T_ref,beta);
		//////////////////////
		//NODES TO PARTICLES//
		//////////////////////
		cout << "check"  << "\n";
		mpmData.nodesToParticlesRESIDUAL(deltaTT,T_sol,T_liq,T_ref,k_sol,k_liq,sigma_y,H,mu,K,beta);
		///UPDATE TIME:
	   	t += deltaTT;
	};

		
};
