#include "camera3d.h"

void Camera3D::update() {
	position = position + velocity;
	velocity = velocity * slowdown;
}

Camera3D::Camera3D() {
	position = glm::vec3();
	pitchAngle = 45;
	yawAngle = 0;
	slowdown = 0.9;
	position.x = -35;
	position.y = -50;
	position.z = -90;
}

Camera3D::Camera3D(glm::vec3 pos, float pitch, float yaw) {
	position = pos;
	pitchAngle = pitch;
	yawAngle = yaw;
	slowdown = 0.9;
}

Camera3D::~Camera3D() {

}
