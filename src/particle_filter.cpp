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
    num_particles = 100;

    default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and psi

    // Set standard deviations for x, y, and theta
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    // Create normal distributions for x, y and theta
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    // Generate particles
    particles.clear();
    for (int i = 0; i < num_particles; ++i) {
        Particle ptc;

        ptc.id = i;
        ptc.x = dist_x(gen);
        ptc.y = dist_y(gen);
        ptc.theta = dist_theta(gen);
        ptc.weight = 0.5;
        ptc.associations.clear();
        ptc.sense_x.clear();
        ptc.sense_y.clear();

        particles.push_back(ptc);
    }

    // Label the flag for initialization as true
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and psi

    // Set standard deviations for x, y, and theta
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    // Create normal distributions for the noises of x, y and theta
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    // Predict particles
    for (int i = 0; i < num_particles; ++i) {

        // The predicted x, y, and theta
        double x_f,y_f,theta_f;

        if (abs(yaw_rate) >= 0.00001) { // theta is not zero
            x_f = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
            y_f = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
            theta_f = particles[i].theta+yaw_rate*delta_t;
        }
        else { // theta is zero
            x_f = particles[i].x + velocity*cos(particles[i].theta)*delta_t;
            y_f = particles[i].y + velocity*sin(particles[i].theta)*delta_t;
            theta_f = particles[i].theta;
        }

        // add noise
        particles[i].x = x_f + dist_x(gen);
        particles[i].y = y_f + dist_y(gen);
        particles[i].theta = theta_f + dist_theta(gen);
    }

    std::cout << "predicted" << endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for(auto obs: observations) {
        // It is assumed that the coordinates of observations is the same as the predicted one
        double distance = 99999999;
        for(auto pre: predicted) {
            if (dist(obs.x,obs.y,pre.x,pre.y) < distance) {
                distance = dist(obs.x,obs.y,pre.x,pre.y);
                obs.id = pre.id;
            }
        }
    }
}

void cal2global(Particle P, std::vector<LandmarkObs> car_cords, std::vector<LandmarkObs>& global_cords) {
    // This function is used to change an array of observations in car coordinates to global coordinates
    // @param P particle
    // @param car_cords observations in car coordinates
    // @param global_cords observations in global coordinates
    global_cords.clear();
    for(auto car_cord: car_cords) {
        LandmarkObs global_cord;
        global_cord.x = P.x + car_cord.x*cos(P.theta) - car_cord.y*sin(P.theta);
        global_cord.y = P.y + car_cord.x*sin(P.theta) + car_cord.y*cos(P.theta);
        global_cord.id = car_cord.id;
        global_cords.push_back(global_cord);
    }
}

void sense(Particle P, Map map_landmarks, double sensor_range, std::vector<LandmarkObs>& predicted) {
    // Get the predicted observations in global coordinates for particle P
    predicted.clear();
    for(auto landmark: map_landmarks.landmark_list) {
        // Check if a landmark can be sensed
        if (dist(P.x,P.y,landmark.x_f,landmark.y_f) <= sensor_range) {
            LandmarkObs mark;
            // add landmark noise
            mark.id = landmark.id_i;
            mark.x = landmark.x_f;
            mark.y = landmark.y_f;
            predicted.push_back(mark);
        }
    }
}

double Gaussian1D(double x, double mu, double sigma) {
    // Calculate Gaussian function
    return 1/(sigma*sqrt(M_PI*2))*exp(-((x-mu)*(x-mu))/(2*sigma*sigma));
}

void nearest_neighbour(LandmarkObs actual_observation, std::vector<LandmarkObs> predicted_observations,
                       LandmarkObs& neighbour) {
    // return the nearest predicted_observation to the actual_observation
    double distance = 99999999;

    for(auto pred_obs: predicted_observations) {
        if (dist(actual_observation.x,actual_observation.y,pred_obs.x,pred_obs.y) < distance) {
            distance = dist(actual_observation.x,actual_observation.y,pred_obs.x,pred_obs.y);
            neighbour.id = pred_obs.id;
            neighbour.x = pred_obs.x;
            neighbour.y = pred_obs.y;
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

    weights.clear();
    for (auto& P: particles) {
        // Convert the coordinates of observations to global ones
        std::vector<LandmarkObs> actual_observations;
        cal2global(P, observations, actual_observations);
        // Calculate predicted observations
        std::vector<LandmarkObs> predicted_observations;
        sense(P, map_landmarks, sensor_range, predicted_observations);
        // Initialize weight
        P.weight = 1.0;
        // update weight
        for (auto actual_observation: actual_observations) {
            // initialize weight
            // Calculate the nearest neighbour
            LandmarkObs neighbour;
            nearest_neighbour(actual_observation, predicted_observations, neighbour);
            // update the weight
            P.weight *= Gaussian1D(neighbour.x, actual_observation.x, std_landmark[0]) *
                        Gaussian1D(neighbour.y, actual_observation.y, std_landmark[1]);
        }
        weights.push_back(P.weight);
    }

    std::cout << "updated" << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());
    std::vector<Particle> particles_new;
    for(int n=0; n<num_particles; ++n) {
        particles_new.push_back(particles[d(gen)]);
    }
    particles = particles_new;

    std::cout << "resampled" << endl;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
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
