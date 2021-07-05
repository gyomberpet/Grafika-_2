//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Gyomber Peter
// Neptun : PIM313
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char* const vertexSource = R"(
	#version 330
	precision highp float;

	uniform vec3 lookat, right, up;
	layout(location = 0) in vec2 normVertexPosition;
		
	out vec3 p;

	void main() {
		p = lookat + right *  normVertexPosition.x + up * normVertexPosition.y;
		gl_Position = vec4(normVertexPosition.x, normVertexPosition.y, 0, 1);
	}
)";

const char * const fragmentSource = R"(
	#version 330
	precision highp float;
	
	struct Hit {
		float t;
		vec3 position, normal;
		int material;
		bool isPortal;
		vec3 portalCenter;
	};

	struct Ray {
		vec3 startPoint, dir, weight;
	};

	const int maxDepth = 7;
	const vec3 La = vec3(0.6f, 0.7f, 0.6f);
	const vec3 lightPos = vec3(0.5, -1, 0.5);
	const vec3 Lout = vec3(1, 1, 1);
	const float epsilon = 0.001f;
	const int sideCount = 12;
	const float d = 0.1f;
	const float r = 0.3f;
	const vec3 center = vec3(0, 0, 0);
	const vec3 trans = vec3(0.0f, 0.0f, -0.18f);
			
	uniform vec3 eye;
	uniform int sides[5 * sideCount];
	uniform vec3 vertices[20];
	uniform vec3 sideCenters[sideCount];
	uniform vec3 F0[2];
	uniform mat4 Q;	
	uniform vec3 ka;
	uniform vec3 kd;
	uniform vec3 ks;
	uniform float shininess;
	uniform float PI;
	

	vec4 quaterniontMult(vec4 q1, vec4 q2) {
		vec3 d1 = vec3(q1.y, q1.z, q1.w);
		vec3 d2 = vec3(q2.y, q2.z, q2.w);
		vec3 d = q1.x * d2 + q2.x * d1 + cross(d1, d2);
		return vec4(q1.x * q2.x - dot(d1, d2), d.x, d.y, d.z);
	}

	vec3 rotate(vec3 point, vec3 d, vec3 pivot, float alpha) {
		vec4 q = vec4(cos(alpha / 2), d.x * sin(alpha / 2), d.y * sin(alpha / 2), d.z * sin(alpha / 2));
		vec4 qCon = vec4(cos(alpha / 2), -d.x * sin(alpha / 2), -d.y * sin(alpha / 2), -d.z * sin(alpha / 2));
		point = point - pivot;
		vec4 p = vec4(0, point.x, point.y, point.z);
		vec4 res = quaterniontMult(quaterniontMult(q, p), qCon);
		return vec3(res.y, res.z, res.w) + pivot;
	}

	vec3 gradient(vec3 pos) {
		vec4 grad = vec4(pos.x, pos.y, pos.z, 1) * Q * 2;
		return vec3(grad.x, grad.y, grad.z);
	}

	Hit intersectGoldObject(Ray ray, Hit hit){
		vec3 start = ray.startPoint - trans;
		vec4 s = vec4(start.x, start.y, start.z, 1);
		vec4 d = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		float a = dot(d * Q, d);
		float b = (dot(s * Q, d)) * 2.0f;
		float c = dot(s * Q, s);
		float disc = b * b - 4.0f * a * c;
		if (disc < 0) return hit;

		float t1 = (-b + sqrt(disc)) / (2.0f * a);
		vec3 p1 = ray.startPoint + ray.dir * t1;
		float distance1 = length(p1 - center);
		if (distance1 > r) t1 = -1;

		float t2 = (-b - sqrt(disc)) / (2.0f * a);
		vec3 p2 = ray.startPoint + ray.dir * t2;
		float distance2 = length(p2 - center);
		if (distance2 > r) t2 = -1;

		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0) hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t1 < t2) hit.t = t1;
		else hit.t = t2;

		hit.position = ray.startPoint + ray.dir * hit.t + trans;
		hit.normal = normalize(gradient(hit.position));
		hit.material = 1;
		return hit;
	}

	vec3 normalVector(int i) {
		vec3 p1 = vertices[sides[i * 5]];
		vec3 p2 = vertices[sides[i * 5 + 1]];
		vec3 p3 = vertices[sides[i * 5 + 2]];
		vec3 normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0) 
			normal = -normal;
		return normalize(normal);
	}

	bool isPort(int i, vec3 interPoint) {
		vec3 cp = sideCenters[i];
		vec3 potalPoints[5];
		for (int j = 0; j < 5; j++) {
			vec3 v = vertices[sides[i * 5 + j]];
			vec3 a = v + normalize(cp - v) * d;
			potalPoints[j] = a;
		}
		vec3 norm = normalVector(i);
		for (int j = 0; j < 5; j++) {
			vec3 pj = potalPoints[j];
			vec3 v;
			if (j < 4) v = potalPoints[j + 1] - pj;
			else v = potalPoints[0] - pj;
			vec3 nj = cross(norm, v);
			if (dot(nj, pj - cp) < 0) nj = -nj;
			if (dot(nj, pj - interPoint) < 0) {
				return false;
			}
		}
		return true;
	}

	Hit intersectDodecahedron(Ray ray, Hit bestHit){
		for (int i = 0; i < sideCount; i++) {
			vec3 p = vertices[sides[i * 5]];
			vec3 normal = normalVector(i);
			float t;
			if (abs(dot(ray.dir, normal)) < epsilon) {
				t = -1;
			}
			else {
				t = dot(p - ray.startPoint, normal) / dot(ray.dir, normal);
			}
			if (t > epsilon && (t < bestHit.t || bestHit.t < 0)) {
				vec3 interPoint = ray.startPoint + ray.dir * t;
				bool inside = true;
				for (int j = 0; j < sideCount; j++) {
					if (i != j) {
						vec3 pj = vertices[sides[j * 5]];
						vec3 nj = normalVector(j);
						if (dot(nj, pj - interPoint) < 0) {
							inside = false;
							break;
						}
					}
				}
				if (inside) {
					bestHit.t = t;
					bestHit.position = interPoint;
					bestHit.normal = normal;	
					if(!isPort(i, interPoint)){	
						bestHit.material = 0;
						bestHit.isPortal = false;
					} else{
						bestHit.material = 2;
						bestHit.isPortal = true;
						bestHit.portalCenter = sideCenters[i];
					}
				}
			}
		}	
		return bestHit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		bestHit.isPortal = false;
		bestHit = intersectGoldObject(ray, bestHit);
		bestHit = intersectDodecahedron(ray, bestHit);
		if (dot(ray.dir, bestHit.normal) > 0)
			bestHit.normal = -bestHit.normal;
		return bestHit;
	}

	vec3 trace(Ray ray) {
		vec3 outRad = vec3(0, 0, 0);
		for(int i = 0; i < maxDepth; i++){
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) break;
			if(hit.material == 0){
				vec3 shadedPoint = hit.position + hit.normal * epsilon;
				vec3 lightDir = normalize(lightPos - shadedPoint);
				Hit shadowHit = firstIntersect(Ray(shadedPoint, lightDir, vec3(0,0,0)));
				float shadowDist = length(shadedPoint - shadowHit.position);
				float lightDist = length(shadedPoint - lightPos);
				float cosTheta = dot(hit.normal, lightDir);
				if (cosTheta > 0 && (shadowHit.t < 0 || shadowDist > lightDist)) {
					vec3 LeIn = Lout / dot(lightPos - hit.position, lightPos - hit.position);
					outRad = outRad + ray.weight *LeIn * kd * cosTheta;
					vec3 halfWay = normalize(-ray.dir + lightDir);
					float cosDelta = dot(hit.normal, halfWay);
					if (cosDelta > 0) {
						outRad = outRad + ray.weight * LeIn * ks * pow(cosDelta, shininess);
					}
				}
				ray.weight = ray.weight * ka;
				break;
			}
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosAlpha = -dot(ray.dir, hit.normal);
			vec3 Frenel = F0[hit.material - 1] + (vec3(1, 1, 1) - F0[hit.material - 1]) * pow(1 - cosAlpha, 5);
			ray.weight = ray.weight * Frenel;
			ray.startPoint = hit.position + hit.normal * epsilon;
			ray.dir = normalize(reflectedDir);
			if(hit.material == 2){
				ray.startPoint = rotate(ray.startPoint, hit.normal, hit.portalCenter, 2 * PI / 5);
				ray.dir = rotate(ray.dir, hit.normal, hit.portalCenter, 2 * PI / 5 );
			}
		}
		outRad = outRad + ray.weight * La;
		return outRad;
	}

	in vec3 p; 
	out vec4 outColor;

	void main() {
		Ray ray;
		ray.startPoint = eye;
		ray.dir = normalize(p - eye);
		ray.weight = vec3(1, 1, 1);
		outColor = vec4(trace(ray), 1);
	}
)";



const vec3 one(1, 1, 1);

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

vec4 quaterniontMult(vec4 q1, vec4 q2) {
	vec3 d1 = vec3(q1.y, q1.z, q1.w);
	vec3 d2 = vec3(q2.y, q2.z, q2.w);
	vec3 d = q1.x * d2 + q2.x * d1 + cross(d1, d2);
	return vec4(q1.x * q2.x - dot(d1, d2), d.x, d.y, d.z);
}

vec3 rotate(vec3 point, vec3 d, vec3 pivot, float alpha) {
	vec4 q = vec4(cos(alpha / 2), d.x * sin(alpha / 2), d.y * sin(alpha / 2), d.z * sin(alpha / 2));
	vec4 qCon = vec4(cos(alpha / 2), -d.x * sin(alpha / 2), -d.y * sin(alpha / 2), -d.z * sin(alpha / 2));
	point = point - pivot;
	vec4 p = vec4(0, point.x, point.y, point.z);
	vec4 res = quaterniontMult(quaterniontMult(q, p), qCon);
	return vec3(res.y, res.z, res.w) + pivot;
}

struct Camera {
	vec3 eye, lookat, right, up;
	float viewAngle;
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _viewAngle) {
		eye = _eye;
		lookat = _lookat;
		viewAngle = _viewAngle;
		vec3 w = lookat - eye;
		float windowSize = length(w) * tanf(viewAngle / 2.0f);
		right = normalize(cross(w, vup)) * windowSize;
		up = normalize(cross(right, w)) * windowSize;
	}

	void Animate(float t) {
		vec3 dir = eye - lookat;
		eye = rotate(eye, vec3(0, 0, 1), lookat, t);
		up = rotate(up, vec3(0, 0, 1), lookat, t);
		set(eye, lookat, up, viewAngle);
	}
};

GPUProgram gpuProgram;
Camera camera;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	unsigned int vao, vbo;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(vertexCoords),
		vertexCoords,
		GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0,
		2, GL_FLOAT, GL_FALSE,
		0, NULL);

	gpuProgram.create(vertexSource, fragmentSource, "outColor");

	vec3 eye = vec3(0.0f, -0.95f, -0.95f);
	vec3 upv = vec3(0, -1, 1);
	vec3 lookat = vec3(0, 0, 0);
	float viewAngle = M_PI / 4;
	camera.set(eye, lookat, upv, viewAngle);

	std::vector<vec3> vertices = {
		vec3(0, 0.618f, 1.618f), vec3(0, -0.618f, 1.618f), vec3(0, -0.618f, -1.618f), vec3(0, 0.618f, -1.618f),
		vec3(1.618f, 0, 0.618f), vec3(-1.618f, 0, 0.618f), vec3(-1.618f, 0, -0.618f), vec3(1.618f, 0, -0.618f),
		vec3(0.618f, 1.618f, 0), vec3(-0.618f, 1.618f, 0), vec3(-0.618f, -1.618f, 0), vec3(0.618f, -1.618f, 0),
		vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, -1, 1), vec3(1, -1, 1),
		vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1)
	};
	for (int i = 0; i < vertices.size(); i++) {
		gpuProgram.setUniform(vertices[i], "vertices[" + std::to_string(i) + "]");
	}
	std::vector<int> sides = {
		0,1,15,4,12,  0,12,8,9,13,  0,13,5,14,1,  1,14,10,11,15,  2,3,17,7,16,  2,16,11,10,19,
		2,19,6,18,3,  18,9,8,17,3,  15,11,16,7,4,  4,7,17,8,12,  13,9,18,6,5,  5,6,19,10,14
	};
	for (int i = 0; i < sides.size(); i++) {
		gpuProgram.setUniform(sides[i], "sides[" + std::to_string(i) + "]");
	}
	for (int i = 0; i < 12; i++) {
		vec3 center(0, 0, 0);
		for (int j = 0; j < 5; j++) {
			center = center + vertices[sides[i * 5 + j]];
		}
		center = center / 5;
		gpuProgram.setUniform(center, "sideCenters[" + std::to_string(i) + "]");
	}

	vec3 n(0.17f, 0.35f, 1.5f);
	vec3 kappa(3.1f, 2.7f, 1.9f);
	vec3 F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	gpuProgram.setUniform(F0, "F0[0]");
	F0 = vec3(1, 1, 1);
	gpuProgram.setUniform(F0, "F0[1]");

	float a = 3;
	float b = 3;
	float c = 2;
	mat4 Q = { a, 0,  0,  0,
			   0, b,  0,  0,
			   0, 0,  0, -c/2,
			   0, 0, -c/2, 0 };
	gpuProgram.setUniform(Q, "Q");

	vec3 kd = vec3(0.3f, 0.2f, 0.2f);
	vec3 ks = vec3(2, 2, 2);
	float shininess = 50;
	gpuProgram.setUniform(kd, "kd");
	gpuProgram.setUniform(kd * M_PI, "ka");
	gpuProgram.setUniform(ks, "ks");
	gpuProgram.setUniform(shininess, "shininess");
	gpuProgram.setUniform(float(M_PI), "PI");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	gpuProgram.setUniform(camera.eye, "eye");
	gpuProgram.setUniform(camera.lookat, "lookat");
	gpuProgram.setUniform(camera.right, "right");
	gpuProgram.setUniform(camera.up, "up");

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

void onIdle() {
	camera.Animate(0.002f);
	glutPostRedisplay();
}
