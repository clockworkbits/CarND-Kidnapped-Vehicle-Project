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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  if (!is_initialized) {
    num_particles = 200;

    default_random_engine random_engine;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
      Particle p;
      p.x = dist_x(random_engine);
      p.y = dist_y(random_engine);
      p.theta = dist_theta(random_engine);
      p.weight = 1.0;
      p.id = i; // TODO: do we need it?

      particles.push_back(p);
      weights.push_back(p.weight);
    }

    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine random_engine;

  for (int i = 0; i < num_particles; i++) {
    double x, y, theta;

    Particle &p = particles[i];

    if (yaw_rate < 0.000001 && yaw_rate > - 0.000001) { // Yaw rate is zero
      x = p.x + velocity * delta_t * cos(p.theta);
      y = p.y + velocity * delta_t * sin(p.theta);
      theta = p.theta;
    } else {
      x = p.x + (velocity/yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      y = p.y + (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      theta = p.theta + yaw_rate * delta_t;
    }

    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);

    p.x = dist_x(random_engine);
    p.y = dist_y(random_engine);
    p.theta = dist_theta(random_engine);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  // I didn't implement this method as it's signature is very confusing to me :(
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

  const double std_x = std_landmark[0];
  const double std_y = std_landmark[1];

  for (int i = 0; i < particles.size(); i++) {

    Particle &p = particles[i];

    // Transform the observations into the map space
    vector<LandmarkObs> map_space_observations;

    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs map_obs;
      LandmarkObs obs = observations[j];

      const double cos_theta = cos(p.theta);
      const double sin_theta = sin(p.theta);

      map_obs.x = obs.x * cos_theta - obs.y * sin_theta + p.x;
      map_obs.y = obs.x * sin_theta + obs.y * cos_theta + p.y;

      map_space_observations.push_back(map_obs);
    }

    vector<int> closest_landmark_ids;

    // Find the closest landmark for each observation
    for (int j = 0; j < map_space_observations.size(); j++) {
      LandmarkObs obs = map_space_observations[j];

      int closest_landmark_id = -1;
      double closest_distance = 0.0;

      for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
        Map::single_landmark_s l = map_landmarks.landmark_list[k];

        const double d = dist(obs.x, obs.y, l.x_f, l.y_f);
        if (closest_landmark_id == -1 || d < closest_distance) {
          closest_landmark_id = l.id_i;
          closest_distance = d;
        }
      }

      closest_landmark_ids.push_back(closest_landmark_id);
    }

    // Update the weight
    double weight = 1.0;

    vector<double> sense_x;
    vector<double> sense_y;
    vector<int> associations;

    for (int j = 0; j < map_space_observations.size(); j++) {
      LandmarkObs obs = map_space_observations[j];
      int closest_landmark_id = closest_landmark_ids[j];

      if (closest_landmark_id != -1) {
        sense_x.push_back(obs.x);
        sense_y.push_back(obs.y);
        associations.push_back(closest_landmark_id);

        particles[i] = SetAssociations(p, associations, sense_x, sense_y);

        Map::single_landmark_s closest_landmark = map_landmarks.landmark_list[closest_landmark_id - 1];

        const double dx = obs.x - closest_landmark.x_f;
        const double dy = obs.y - closest_landmark.y_f;

        const long double prob =
            exp(-(dx * dx / (2.0 * std_x * std_x)) - (dy * dy / (2.0 * std_y * std_y))) / (2.0 * M_PI * std_x * std_y);

        if (prob > 0.0) {
          weight *= prob;
        }
      } else {
        cout << "negative landmark :(" << endl;
      }
    }

    if (weight == 0.0) {
      weight = 1.0;
    }

    p.weight = weight;
    weights[i] = p.weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine random_engine;

  discrete_distribution<int> distribution(weights.begin(), weights.end());

  vector<Particle> resampled_particles;

  for (int i = 0; i < num_particles; i++) {
    resampled_particles.push_back(particles[distribution(random_engine)]);
  }

  particles = resampled_particles;
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
