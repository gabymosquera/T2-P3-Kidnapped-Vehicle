 /*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 10;

	default_random_engine gen;
	normal_distribution<double> N_x(x, std[0]); //Random Gaussian Noise X
	normal_distribution<double> N_y(y, std[1]); //Random Gaussian Noise Y
	normal_distribution<double> N_theta(theta, std[2]); //Random Gaussian Noise Theta

	for (int i = 0; i < num_particles; i++)
	{
		
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(1);

	}
	
	is_initialized = true;
	//cout << "Initialization Completed" << endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++)
	{
		double predicted_x;
		double predicted_y;
		double predicted_theta;

		if (yaw_rate == 0)
		{
			predicted_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
			predicted_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
			predicted_theta = particles[i].theta;
		}
		else
		{
			predicted_x = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
			predicted_y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
			predicted_theta = particles[i].theta + yaw_rate*delta_t;

		}

		normal_distribution<double> N_x(predicted_x, std_pos[0]);
		normal_distribution<double> N_y(predicted_y, std_pos[1]);
		normal_distribution<double> N_theta(predicted_theta, std_pos[2]);

		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);

	}

	
	//cout << "Prediction Completed" << endl;

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
		for (int o = 0; o < observations.size(); o++) 
		{
			double closest_distance = distance(predicted[0], observations[o]);
			observations[o].id = predicted[0].id;

			for (int p = 0; p < predicted.size(); p++) 
			{
				//cout << "-------------------COMPARE OBSERVATION WITH LANDMARKS-------------------" << endl;
				double current_distance = distance(predicted[p], observations[o]);
				
				if (current_distance < closest_distance) 
				{
					//cout << "--------------UPDATING DISTANCES & IDs--------------" << endl;
					closest_distance = current_distance;
					observations[o].id = predicted[p].id;

				}
			}

		//cout << "For Transformed Observation # " << o <<  " (x, y): (" << observations[o].x << ", " << observations[o].y << ")" << endl;
		//cout << "Closest Landmark Location id# " << predicted[(observations[o].id)-1].id << " (x, y): (" << predicted[(observations[o].id)-1].x << ", " << predicted[(observations[o].id)-1].y << ")" << endl;
		}
}

double ParticleFilter::distance(LandmarkObs o1, LandmarkObs o2) {
	return sqrt(pow(o2.x-o1.x, 2) + pow(o2.y-o1.y, 2));
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sigma_x = std_landmark[0];
	//cout << "sigma x = " << sigma_x << endl;
	double sigma_y = std_landmark[1];
	//cout << "sigma y = " << sigma_y << endl;
	double cov_x = pow(sigma_x, 2.0);
	//cout << "cov x = " << cov_x << endl;
	double cov_y = pow(sigma_y, 2.0);
	//cout << "cov y = " << cov_y << endl;
	double multiplier = 1.0/(2*M_PI*sigma_x*sigma_y);
	//cout << "multiplier = " << multiplier << endl;


	//cout << "Size of Observations: " << observations.size() << endl;
	//cout << "Size of Particles: " << particles.size() << endl;

	for (int p = 0; p < particles.size(); p++)
	{
		//cout << "---------------------PARTICLES LOOP---------------------" << endl;
		Particle particle = particles[p];
		
		vector<LandmarkObs> trans_observations;
		for (int i = 0; i < observations.size(); i++)
		{
			//cout << "---------------------OBSERVATIONS LOOP---------------------" << endl;
			LandmarkObs trans_obs;
			LandmarkObs obs = observations[i];

			//space transformation from vehicle to map coordinates
			trans_obs.id = 0; //DIFF
			trans_obs.x = (obs.x*cos(particle.theta)-obs.y*sin(particle.theta))+particle.x;
			trans_obs.y = (obs.x*sin(particle.theta)+obs.y*cos(particle.theta))+particle.y;
			trans_observations.push_back(trans_obs);

			//cout << "Particle id =                      " << particle.id << "     Particle Location (x, y): (" << particle.x << ", " << particle.y << ") And Theta: " << particle.theta << endl;
			//cout << "Observed Particle id =             " << obs.id << "      Observed Particle Location (x, y): (" << obs.x << ", " << obs.y << ")" << endl;
			//cout << "Transformed Observed Particle id = " << trans_obs.id << "      Transformed Observed Particle Location (x, y): (" << trans_obs.x << ", " << trans_obs.y << ")" << endl;
		}
	//}

	vector<LandmarkObs> landmark_loc;
	map<int, LandmarkObs> predicted_dict;
	for (int g = 0; g < map_landmarks.landmark_list.size(); g++)
	{
		//cout << "---------------------LANDMARKS LOOP---------------------" << endl;
		LandmarkObs pred;
		//landm = map_landmarks.landmark_list[g];

		pred.id = map_landmarks.landmark_list[g].id_i;
		pred.x = map_landmarks.landmark_list[g].x_f;
		pred.y = map_landmarks.landmark_list[g].y_f;

		//cout << "Landmark id = " << pred.id << "      Landmark x = " << pred.x << "      Landmark y = " << pred.y << endl;

		if (dist(pred.x, pred.y, particle.x, particle.y) > sensor_range) 
		{
			continue; // drop landmark - past the sensor range
		}

		landmark_loc.push_back(pred);
		predicted_dict[pred.id] = pred;
	}

//*********************** DATA ASSOCIATION BEGINS ***********************

		dataAssociation(landmark_loc, trans_observations);
		
		for (int tt = 0; tt < trans_observations.size(); tt++)
		{
			LandmarkObs obs = trans_observations[tt];
			LandmarkObs prediction = predicted_dict[obs.id];
		}
//*********************** DATA ASSOCIATION ENDS ***********************

		vector<int> associations(trans_observations.size());
		vector<double> sense_x(trans_observations.size());
		vector<double> sense_y(trans_observations.size());
		vector<double> new_weights;
		for (int tt = 0; tt < trans_observations.size(); tt++)
		{
			LandmarkObs obs = trans_observations[tt];
			LandmarkObs prediction = predicted_dict[obs.id];

			double obsx = obs.x;
			double obsy = obs.y;
			double mux = prediction.x;
			double muy = prediction.y;

			associations[tt] = trans_observations[tt].id;
			sense_x[tt] = obsx;
			sense_y[tt] = obsy;

			//cout << "Multiplier: " << multiplier << endl;
			double exponent = (pow(obsx - mux, 2.0))/(2.0*cov_x) + (pow(obsy - muy, 2.0))/(2.0*cov_y);
			//cout << "Exponent: " << exponent << endl;
			double partial_weight = multiplier * exp(-exponent);
			//cout << "Partial Weight (multiplier*e^(-exponent): " << partial_weight << endl;

		//cout << "For Transformed Observation # " << tt <<  " (x, y): (" << obs.x << ", " << obs.y << ")" << endl;
		//cout << "Closest Landmark Location id# " << prediction.id << " (x, y): (" << prediction.x << ", " << prediction.y << ")" << endl;

		new_weights.push_back(partial_weight);
		}
		
		double weight = accumulate(new_weights.begin(), new_weights.end(), 1.0, multiplies<double>());

		SetAssociations(particle, associations, sense_x, sense_y);
		particles[p].weight = weight;
		//cout << "Particle " << p << " Weight:"  << particles[p].weight << endl;
		weights[p] = weight;
		//cout << "Weight: " << weights[p] << endl;
	}
	
	//cout << "Weights Completed" << endl;
	 
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> resample_particles;

	for (int i = 0; i < num_particles; i++)
	{
		//cout << "For particle " << i << " Weight before resampling ="<< particles[i].weight << endl;
		resample_particles.push_back(particles[distribution(gen)]);
		//cout << "For particle " << i << " Weight after resampling ="<< resample_particles[i].weight << endl;
	}

	particles = resample_particles;

	//cout << "Resampling Completed" << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;

}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
