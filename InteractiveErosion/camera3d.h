#pragma once
#include <glm/glm.hpp>

class Camera3D
{
public:
	float pitchAngle;
	float yawAngle;
	float slowdown;
	glm::vec3 position;
	glm::vec3 velocity;
	void update();
	Camera3D();
	Camera3D(glm::vec3 position, float pitch, float yaw);
	~Camera3D();
};