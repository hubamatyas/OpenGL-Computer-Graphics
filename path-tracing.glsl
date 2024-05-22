function setup()
{
	UI = {};
	UI.tabs = [];
	UI.titleLong = 'Path Tracer';
	UI.titleShort = 'PathTracer';
	UI.numFrames = 1000;
	UI.maxFPS = 1000;
	UI.renderWidth = 1024;
	UI.renderHeight = 512;

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `Raytracing`,
		id: `TraceFS`,
		initialValue: `#define SOLUTION_LIGHT
#define SOLUTION_BOUNCE
#define SOLUTION_THROUGHPUT
#define SOLUTION_AA
#define SOLUTION_MB
#define SOLUTION_VR

// Turning on SOLUTION_VR will have no effect on its own. Turn on the three options below one-by-one to see the effect of each variance reduction technique. You can turn on multiple techniques at the same time for better results.
#define HALTON
#define IMPORTANCE_SAMPLING
//#define NEXT_EVENT


precision highp float;

#define M_PI 3.14159265359

struct Material {
	#ifdef SOLUTION_LIGHT
	// By changing the Gamma value we can adjust the brightness of each shape in our scence, and therefore change the lighting of the final output image. The Gamma value also controls the contrast of the image, for example a lower value intensifies the direct light on objects which results in higher luminance and consequently a higher contrasted image.
	vec3 emission;
	#endif
	vec3 diffuse;
	vec3 specular;
	float glossiness;
};

struct Sphere {
	vec3 position;
#ifdef SOLUTION_MB
	vec3 motion;
#endif
	float radius;
	Material material;
};

struct Plane {
	vec3 normal;
	float d;
	Material material;
};

const int sphereCount = 4;
const int planeCount = 4;
const int emittingSphereCount = 2;
#ifdef SOLUTION_BOUNCE
const int maxPathLength = 2;
#else
const int maxPathLength = 1;
#endif 

struct Scene {
	Sphere[sphereCount] spheres;
	Plane[planeCount] planes;
};

struct Ray {
	vec3 origin;
	vec3 direction;
};

// Contains all information pertaining to a ray/object intersection
struct HitInfo {
	bool hit;
	float t;
	vec3 position;
	vec3 normal;
	Material material;
};

// Contains info to sample a direction and this directions probability
struct DirectionSample {
	vec3 direction;
	float probability;
};

HitInfo getEmptyHit() {
	Material emptyMaterial;
	#ifdef SOLUTION_LIGHT
	emptyMaterial.emission = vec3(0.0);
	#endif
	emptyMaterial.diffuse = vec3(0.0);
	emptyMaterial.specular = vec3(0.0);
	emptyMaterial.glossiness = 1.0;
	return HitInfo(false, 0.0, vec3(0.0), vec3(0.0), emptyMaterial);
}

// Sorts the two t values such that t1 is smaller than t2
void sortT(inout float t1, inout float t2) {
	// Make t1 the smaller t
	if(t2 < t1)  {
		float temp = t1;
		t1 = t2;
		t2 = temp;
	}
}

// Tests if t is in an interval
bool isTInInterval(const float t, const float tMin, const float tMax) {
	return t > tMin && t < tMax;
}

// Get the smallest t in an interval
bool getSmallestTInInterval(float t0, float t1, const float tMin, const float tMax, inout float smallestTInInterval) {

	sortT(t0, t1);

	// As t0 is smaller, test this first
	if(isTInInterval(t0, tMin, tMax)) {
		smallestTInInterval = t0;
		return true;
	}

	// If t0 was not in the interval, still t1 could be
	if(isTInInterval(t1, tMin, tMax)) {
		smallestTInInterval = t1;
		return true;
	}

	// None was
	return false;
}

// Converts a random integer in 15 bits to a float in (0, 1)
float randomIntegerToRandomFloat(int i) {
	return float(i) / 32768.0;
}

// Returns a random integer for every pixel and dimension that remains the same in all iterations
int pixelIntegerSeed(const int dimensionIndex) {
	vec3 p = vec3(gl_FragCoord.xy, dimensionIndex);
	vec3 r = vec3(23.14069263277926, 2.665144142690225,7.358926345 );
	return int(32768.0 * fract(cos(dot(p,r)) * 123456.0));
}

// Returns a random float for every pixel that remains the same in all iterations
float pixelSeed(const int dimensionIndex) {
	return randomIntegerToRandomFloat(pixelIntegerSeed(dimensionIndex));
}

// The global random seed of this iteration
// It will be set to a new random value in each step
uniform int globalSeed;
int randomSeed;
void initRandomSequence() {
	randomSeed = globalSeed + pixelIntegerSeed(0);
}

// Computes integer  x modulo y not available in most WEBGL SL implementations
int mod(const int x, const int y) {
	return int(float(x) - floor(float(x) / float(y)) * float(y));
}

// Returns the next integer in a pseudo-random sequence
int rand() {
	randomSeed = randomSeed * 1103515245 + 12345;
	return mod(randomSeed / 65536, 32768);
}

float uniformRandomImproved(vec2 co){
    float a = 12.9898;
    float b = 78.233;
    float c = 43758.5453;
    float dt= dot(co.xy ,vec2(a,b));
    float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}

// Returns the next float in this pixels pseudo-random sequence
float uniformRandom() {
	return randomIntegerToRandomFloat(rand());
}

// This is the index of the sample controlled by the framework.
// It increments by one in every call of this shader
uniform int baseSampleIndex;

#if defined(SOLUTION_VR) && defined(HALTON)
// First variance reduction technique is to use the Halton sequence for sampling, also known as a type Quasi-Monte Carlo technique. The loop inside the function calculates the n-th Halton number using the radical inverse base. The Halton sequence is an extension of the Van der Corput sequence which originally uses base 2 to generate a low discrepancy sequence. Used resources at http://www.goddardconsulting.ca/matlab-monte-carlo-assetpaths-halton.html to construct the Halton algorithm.
float haltonSequence(float base, int sampleIndex) {
	float n = float(sampleIndex);
	float f = 1.0;
	float halton = 0.0;

	for (int i = 0; i < 100; i++) {
		if (n <= 0.0) return halton;
		
		f /= base;
		halton += f * mod(n, base);
		n = floor(n / base);
	}

	return halton;
}

#endif

// Returns a well-distributed number in (0,1) for the dimension dimensionIndex
float sample(const int dimensionIndex) {
	#if defined(SOLUTION_VR) && defined(HALTON)
	// Using primes as the base for the sequence which is the main difference compared to the more standard Van der Corput sequence. Using a different prime for each dimension allows us to extend the sequence to multiple dimensions. For example in 2D, we would take 2 and 3 as our bases (the first two primes). Ideallly, a different prime would be assigned to all different dimensions, but, for ease of implementation, we use the first 6 primes here.
	int base = 2;
	if (dimensionIndex == 1) base = 3;
	if (dimensionIndex == 2) base = 5;
	if (dimensionIndex == 3) base = 7;
	if (dimensionIndex == 4) base = 11;
	if (dimensionIndex == 5) base = 13;
	
	// A limitation of the Halton sequence is that it assigns the same random number to all pixels in a given dimension. This introduces structured artefacts, a known side-effect of Quasi-Monte Carlo. To counteract the structured artefacts, we can use the Cranley-Patterson rotation which first adds a random offset to the sequence, then takes the modulo 1 of the result to get the decimal values. These two steps shift the sampled points to achieve a randomised low-discrepancy sequence which is evenly distributed, deterministic and works in multi-dimensions. Cranley-Patterson rotation implemented as explained here: https://nvlabs.github.io/fermat/group__cp__rotations.html
	float halton = haltonSequence(float(base), baseSampleIndex);
	float offset = pixelSeed(dimensionIndex);
	return mod(halton + offset, 1.0);
	#else
	// combining 2 PRNGs to avoid the patterns in the C-standard LCG
	return uniformRandomImproved(vec2(uniformRandom(), uniformRandom()));
	#endif
}

// This is a helper function to sample two-dimensionaly in dimension dimensionIndex
vec2 sample2(const int dimensionIndex) {
	return vec2(sample(dimensionIndex + 0), sample(dimensionIndex + 1));
}

vec3 sample3(const int dimensionIndex) {
	return vec3(sample(dimensionIndex + 0), sample(dimensionIndex + 1), sample(dimensionIndex + 2));
}

// This is a register of all dimensions that we will want to sample.
// Thanks to Iliyan Georgiev from Solid Angle for explaining proper housekeeping of sample dimensions in ranomdized Quasi-Monte Carlo
//
// There are infinitely many path sampling dimensions.
// These start at PATH_SAMPLE_DIMENSION.
// The 2D sample pair for vertex i is at PATH_SAMPLE_DIMENSION + PATH_SAMPLE_DIMENSION_MULTIPLIER * i + 0
#define ANTI_ALIAS_SAMPLE_DIMENSION 0
#define TIME_SAMPLE_DIMENSION 1
#define PATH_SAMPLE_DIMENSION 3

// This is 2 for two dimensions and 2 as we use it for two purposese: NEE and path connection
#define PATH_SAMPLE_DIMENSION_MULTIPLIER (2 * 2)

vec3 getEmission(const Material material, const vec3 normal) {
	#ifdef SOLUTION_LIGHT
	return material.emission;
	#else
	// This is wrong. It just returns the diffuse color so that you see something to be sure it is working.
	return material.diffuse;
	#endif
}

vec3 getReflectance(const Material material, const vec3 normal, const vec3 inDirection, const vec3 outDirection) {
	#ifdef SOLUTION_THROUGHPUT
	// Calculate the BRDF term (ie. reflectivity) of the surface at a given point. To do so, we first use 
	// the reflect() function in GLSL to calculate the reflection direction of the incoming ray given the
	// normal of the surface that was hit. We then use the physically-correct Phong BRDF formula given in
	// the coursework brief to calculate the specular and diffuse terms and add them together. The energy
	// term ensures that we preserve energy for all values of glossiness. This energy term, in particular
	// the normalisation factor (n+2)/2π is what makes the Phong formula physically-correct.
	vec3 reflectedDirection = reflect(inDirection, normal);
	
	float specularTerm  = pow(max(0.0, dot(outDirection, reflectedDirection)), material.glossiness);
	float energyTerm = (material.glossiness + 2.0) / (2.0 * M_PI);
	vec3 specular = material.specular * energyTerm * specularTerm;
	
	vec3 diffuse = material.diffuse / M_PI;
	return diffuse + specular;
	#else
	return vec3(1.0);
	#endif
}

vec3 getGeometricTerm(const Material material, const vec3 normal, const vec3 inDirection, const vec3 outDirection) {
	#ifdef SOLUTION_THROUGHPUT
	// Calculate the cosine of the angle between the and normal of the point we hit and the
	// randomly sampled direction (ie. outDirection). cos(theta) = (a.b) as we know both
	// outDirection and normal are already normalised. cos(theta) will serve as the geometric
	// term in the rendering equation in intersectScene()
	float cosTheta = dot(outDirection, normal);
	
	// Clamp to 0 to avoid negative cos values
	return vec3(max(0.0, cosTheta));
	#else
	return vec3(1.0);
	#endif
}

vec3 sphericalToEuclidean(float theta, float phi) {
	float x = sin(theta) * cos(phi);
	float y = sin(theta) * sin(phi);
	float z = cos(theta);
	return vec3(x, y, z);	
}

// New function defined for Question 2. SOLUTION_BOUNCE to separte the logical parts
// of getRandomDirection
vec2 getRandomSphericalCoordinates(const int dimensionIndex) {
	float theta = acos(2.0 * sample(dimensionIndex) - 1.0);
	float phi = sample(dimensionIndex + 1) * 2.0 * M_PI;
	return vec2(theta, phi);
}

vec3 getRandomDirection(const int dimensionIndex) {
	#ifdef SOLUTION_BOUNCE
	// 1st part of logic - get spherical sample coordinates, theta and phi, from a uniform
	// distribution. getRandomSphericalCoordinates() above was defined and given a proper
	// name to separate the first logical part of getRandomDirection()
	vec2 sampleCoord = getRandomSphericalCoordinates(dimensionIndex);
	
	// 2nd part of logic - convert the spherical sample coordinates to Cartesian (ie. Euclidean)
	// coordinates (x, y, z) in 3D space. 
	return sphericalToEuclidean(sampleCoord.x, sampleCoord.y);
	#else
	// Put your code to compute a random direction in 3D in the #ifdef above
	return vec3(0);
	#endif
}


HitInfo intersectSphere(const Ray ray, Sphere sphere, const float tMin, const float tMax) {

#ifdef SOLUTION_MB
	sphere.position += sphere.motion * sample(TIME_SAMPLE_DIMENSION);
#endif
	
	vec3 to_sphere = ray.origin - sphere.position;

	float a = dot(ray.direction, ray.direction);
	float b = 2.0 * dot(ray.direction, to_sphere);
	float c = dot(to_sphere, to_sphere) - sphere.radius * sphere.radius;
	float D = b * b - 4.0 * a * c;
	if (D > 0.0)
	{
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);

		float smallestTInInterval;
		if(!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
			return getEmptyHit();
		}

		vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;

		vec3 normal =
			length(ray.origin - sphere.position) < sphere.radius + 0.001?
			-normalize(hitPosition - sphere.position) :
		normalize(hitPosition - sphere.position);

		return HitInfo(
			true,
			smallestTInInterval,
			hitPosition,
			normal,
			sphere.material);
	}
	return getEmptyHit();
}

HitInfo intersectPlane(Ray ray, Plane plane) {
	float t = -(dot(ray.origin, plane.normal) + plane.d) / dot(ray.direction, plane.normal);
	vec3 hitPosition = ray.origin + t * ray.direction;
	return HitInfo(
		true,
		t,
		hitPosition,
		normalize(plane.normal),
		plane.material);
	return getEmptyHit();
}

float lengthSquared(const vec3 x) {
	return dot(x, x);
}

HitInfo intersectScene(Scene scene, Ray ray, const float tMin, const float tMax)
{
	HitInfo best_hit_info;
	best_hit_info.t = tMax;
	best_hit_info.hit = false;

	for (int i = 0; i < sphereCount; ++i) {
		Sphere sphere = scene.spheres[i];
		HitInfo hit_info = intersectSphere(ray, sphere, tMin, tMax);

		if(	hit_info.hit &&
		   hit_info.t < best_hit_info.t &&
		   hit_info.t > tMin)
		{
			best_hit_info = hit_info;
		}
	}

	for (int i = 0; i < planeCount; ++i) {
		Plane plane = scene.planes[i];
		HitInfo hit_info = intersectPlane(ray, plane);

		if(	hit_info.hit &&
		   hit_info.t < best_hit_info.t &&
		   hit_info.t > tMin)
		{
			best_hit_info = hit_info;
		}
	}

	return best_hit_info;
}

mat3 transpose(mat3 m) {
	return mat3(
		m[0][0], m[1][0], m[2][0],
		m[0][1], m[1][1], m[2][1],
		m[0][2], m[1][2], m[2][2]
	);
}

// This function creates a matrix to transform from global space into a local space oriented around the provided vector.
mat3 makeLocalFrame(const vec3 vector) {
	#ifdef SOLUTION_VR
	// Use the Gram-Schmidt process to carry out orthogonal projection (ie. construct an orthonormal basis for the local space). In our case, orthogonal projection works just as fine as perspective projection would but it's signicantly easier to implement.
	
	vec3 w = normalize(vector);
	// Find an arbitrary non-parallel vector to w
	vec3 nonParallel = abs(w.x) > 0.1 ? vec3(0, 1, 0) : vec3(1, 0, 0);
	// Create an orthogonal vector to w using the cross product (ie. Gram-Schmidt process)
	vec3 u = normalize(cross(nonParallel, w));
	// Identify the third ortogonal vector
	vec3 v = cross(w, u);

	return mat3(u, v, w);
	#else
	return mat3(1.0);
	#endif
}

float luminance(const vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

#define EPSILON (1e-6)  // for avoiding numeric issues caused by floating point accuracy

#if defined(SOLUTION_VR) && defined(NEXT_EVENT)
// Gets the direct light contribution at a given point for Next Event Estimation purposes. NEE combines light source sampling and BRDF sampling to reduce variance and speed up convergence. As with NEE we always make a connection to the light sources our image will render light even in areas where rays might never reach due to chance.
vec3 getDirectLight(const Scene scene, HitInfo hitInfo, Ray incomingRay, int dimensionIndex) {
	vec3 directLight;
	
	// Iterate over each sphere to find the light sources. In our case, we know that only spheres emmit light. For more general purposes, planes and other shapes should be included too to identify light sources.
	for (int i = 0; i < sphereCount; i++) {
		Sphere light = scene.spheres[i];
		// If sphere is not a light source there's no direct light to capture, so skip
		if (getEmission(light.material, hitInfo.normal) == vec3(0.0)) continue;
		
		// Compute the direction from the hit point to the light source. Make sure to sample a truly random point (ie. getRandomDirection) on the sphere to maintain an unbiased estimator
		vec3 randomSampleOnSphere = light.position + light.radius * getRandomDirection(dimensionIndex);
		vec3 hitToLight = randomSampleOnSphere - hitInfo.position;
		vec3 lightDirection = normalize(hitToLight);

		// Compute the reflected light at the original hit point but with the outgoing direction as the direction pointing at the light
		vec3 geometricTerm = getGeometricTerm(hitInfo.material, hitInfo.normal, -incomingRay.direction, lightDirection);
		vec3 reflectance = getReflectance(hitInfo.material, hitInfo.normal, -incomingRay.direction, lightDirection);
		
		// Sum direct light contribution from each light source
		directLight += reflectance * geometricTerm;
	}
	
	return directLight;
}
#endif

DirectionSample sampleDirection(const vec3 normal, const vec3 inDirection, const vec3 diffuse, const vec3 specular, const float n, const int dimensionIndex) {
	DirectionSample result;
		
	#if defined(SOLUTION_VR) && defined(IMPORTANCE_SAMPLING)	

	// Using cosine-weighted distribution on the hemisphere to implement Importance Sampling for the geometric term. https://ameye.dev/notes/sampling-the-hemisphere/
	float theta = acos(sqrt(1.0 - sample(dimensionIndex)));
	float phi = 2.0 * M_PI * sample(dimensionIndex + 1);
	
	vec3 localDirection = sphericalToEuclidean(theta, phi);

	// Transform local direction to global space
	result.direction = makeLocalFrame(normal) * localDirection;


	// Compute the probability of the sampled direction (cosine-weighted). This ensures that we assing a higher probability to paths where radiance is higher. PDF is 1/PI * cos which we have to divide by to get unbiased results
	result.probability = cos(theta) / M_PI;
	#else
	// Depending on the technique: put your variance reduction code in the #ifdef above 
	result.direction = getRandomDirection(dimensionIndex);	
	result.probability = 1.0 / (4.0 * M_PI);
	#endif
	return result;
}

vec3 unitTest(HitInfo hitInfo, Ray incomingRay, int i) {
	DirectionSample directionSample = sampleDirection(hitInfo.normal, incomingRay.direction, hitInfo.material.diffuse, hitInfo.material.specular, 0.0, PATH_SAMPLE_DIMENSION+2*i);
	return directionSample.direction;
}

vec3 samplePath(const Scene scene, const Ray initialRay) {

	// Initial result is black
	vec3 result = vec3(0);

	Ray incomingRay = initialRay;
	vec3 throughput = vec3(1.0);
	for(int i = 0; i < maxPathLength; i++) {
		HitInfo hitInfo = intersectScene(scene, incomingRay, 0.001, 10000.0);

		if(!hitInfo.hit) return result;

		result += throughput * getEmission(hitInfo.material, hitInfo.normal);
		
		Ray outgoingRay;
		DirectionSample directionSample;
		#ifdef SOLUTION_BOUNCE
		
		// A third function that the unit test could involve is sampleDirection(). This way, with SOLUTION_VR disabled,
		// the unit test tests the randomness of the vector returned by getRandomDirection. With SOLUTION_VR enabled
		// the randomness of getRandomDirection() as well as the effect of variance reduction techniques on the
		// randomness of the vector returned by getRandomDirection() is tested.
		
		// Randomness of sample direction can be tested if we think of the direction as a colour. From sphericalToEuclidean()
		// we know that Cartesian coordianates are in (-1, 1) because we're using sin and cos to convert from spherical coordinates.
		// Therefore, if the directions are truly random we can expect the average to converge to 0 and the output image to be black.
		
		// Set testFlag to true to run the test
		bool testFlag = false;
		if (testFlag == true) {
			return unitTest(hitInfo, incomingRay, i);
		}
		
		directionSample = sampleDirection(hitInfo.normal, incomingRay.direction, hitInfo.material.diffuse, hitInfo.material.specular, hitInfo.material.glossiness, PATH_SAMPLE_DIMENSION+2*i);
		outgoingRay.origin = hitInfo.position;
		outgoingRay.direction = directionSample.direction;
		
		#else
			// Put your code to compute the next ray in the #ifdef above
		#endif
		
		#ifdef SOLUTION_THROUGHPUT
		// Multiply the reflectance (ie. physically-correct Phong BRDF) term and geometric term as specified in the rendering equation to compute the throughput of every light path vertex to camera.
		
		vec3 geometricTerm = getGeometricTerm(hitInfo.material, hitInfo.normal, incomingRay.direction, outgoingRay.direction);
		vec3 reflectance = getReflectance(hitInfo.material, hitInfo.normal, incomingRay.direction, outgoingRay.direction);
		throughput *= reflectance * geometricTerm;
	
		#else
		// Compute the proper throughput in the #ifdef above 
		throughput *= 0.1;
		#endif

		// div by probability of sampled direction 
		throughput /= directionSample.probability;
		
		#if defined(SOLUTION_VR) && defined(NEXT_EVENT)
		// Add contribution from direct lightsources to the current throughput. No need to divide by probability because we always want the contribution of direct light with Next Event Estimation
		vec3 directLight = getDirectLight(scene, hitInfo, incomingRay, PATH_SAMPLE_DIMENSION+2*i);
		throughput += directLight;
		// Divide by 2 as now we are adding on double the amount of light to the throughput
		throughput /= 2.0;
		#endif
	
		#ifdef SOLUTION_BOUNCE
		incomingRay = outgoingRay;
		#else
		// Put some handling of the next and the current ray in the #ifdef above
		#endif
	}
	return result;
}

uniform ivec2 resolution;
Ray getFragCoordRay(const vec2 fragCoord) {

	float sensorDistance = 1.0;
	vec3 origin = vec3(0, 0, sensorDistance);
	vec2 sensorMin = vec2(-1, -0.5);
	vec2 sensorMax = vec2(1, 0.5);
	vec2 pixelSize = (sensorMax - sensorMin) / vec2(resolution);
	vec3 direction = normalize(vec3(sensorMin + pixelSize * fragCoord, -sensorDistance));

	float apertureSize = 0.0;
	float focalPlane = 100.0;
	vec3 sensorPosition = origin + focalPlane * direction;
	origin.xy += -vec2(0.5);
	direction = normalize(sensorPosition - origin);

	return Ray(origin, direction);
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {
	initRandomSequence();

	#ifdef SOLUTION_AA
	// Use a 1x1 box filter to achieve anti-aliasing. sample2() returns a vec2 with values in [0, 1] and is used to send a ray to a random point inside the pixel (ie. 1x1 box). This random offset ensures that we never sample regularly. Sampling regularly can result in aliasing, hence by avoiding it with randomness we get smoother and blurred edges. vec2(0.5) is necessary to centre the random point insdie the current pixel. The effect of anti-aliasing can be increased by either increasing the box filter size (filterSize) or the number of random rays sent out (randomRays). Keep in mind that these will have a negative effect on performance.
	vec2 sampleCoord = fragCoord;
	const float filterSize = 1.0;
	const int randomRays = 1;
	
	for(int i = 0; i < randomRays; i++) {
		sampleCoord += filterSize * sample2(ANTI_ALIAS_SAMPLE_DIMENSION) - filterSize * vec2(0.5);
	}
	
	#else  	
	// Put your anti-aliasing code in the #ifdef above
	vec2 sampleCoord = fragCoord;
	#endif
	return samplePath(scene, getFragCoordRay(sampleCoord));
}

void loadScene1(inout Scene scene) {

	scene.spheres[0].position = vec3(7, -2, -12);
	scene.spheres[0].radius = 2.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT
	scene.spheres[0].material.emission = 15.0 * vec3(0.9, 0.9, 0.5);
#endif
	scene.spheres[0].material.diffuse = vec3(0.5);
	scene.spheres[0].material.specular = vec3(0.5);
	scene.spheres[0].material.glossiness = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
	scene.spheres[0].motion = vec3(0.0);
#endif
	
	scene.spheres[1].position = vec3(-8, 4, -13);
	scene.spheres[1].radius = 1.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT  
	scene.spheres[1].material.emission = 15.0 * vec3(0.8, 0.3, 0.1);
#endif
	scene.spheres[1].material.diffuse = vec3(0.5);
	scene.spheres[1].material.specular = vec3(0.5);
	scene.spheres[1].material.glossiness = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
	scene.spheres[1].motion = vec3(0.0);
#endif
	
	scene.spheres[2].position = vec3(-2, -2, -12);
	scene.spheres[2].radius = 3.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT  
	scene.spheres[2].material.emission = vec3(0.0);
#endif  
	scene.spheres[2].material.diffuse = vec3(0.2, 0.5, 0.8);
	scene.spheres[2].material.specular = vec3(0.8);
	scene.spheres[2].material.glossiness = 40.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
	scene.spheres[2].motion = vec3(-3.0, 0.0, 3.0);
#endif
	
	scene.spheres[3].position = vec3(3, -3.5, -14);
	scene.spheres[3].radius = 1.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT  
	scene.spheres[3].material.emission = vec3(0.0);
#endif  
	scene.spheres[3].material.diffuse = vec3(0.9, 0.8, 0.8);
	scene.spheres[3].material.specular = vec3(1.0);
	scene.spheres[3].material.glossiness = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
	scene.spheres[3].motion = vec3(2.0, 4.0, 1.0);
#endif
	
	scene.planes[0].normal = vec3(0, 1, 0);
	scene.planes[0].d = 4.5;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT    
	scene.planes[0].material.emission = vec3(0.0);
#endif
	scene.planes[0].material.diffuse = vec3(0.8);
	scene.planes[0].material.specular = vec3(0.0);
	scene.planes[0].material.glossiness = 50.0;    

	scene.planes[1].normal = vec3(0, 0, 1);
	scene.planes[1].d = 18.5;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT  
	scene.planes[1].material.emission = vec3(0.0);
#endif
	scene.planes[1].material.diffuse = vec3(0.9, 0.6, 0.3);
	scene.planes[1].material.specular = vec3(0.02);
	scene.planes[1].material.glossiness = 3000.0;

	scene.planes[2].normal = vec3(1, 0,0);
	scene.planes[2].d = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT    
	scene.planes[2].material.emission = vec3(0.0);
#endif
	
	scene.planes[2].material.diffuse = vec3(0.2);
	scene.planes[2].material.specular = vec3(0.1);
	scene.planes[2].material.glossiness = 100.0; 

	scene.planes[3].normal = vec3(-1, 0,0);
	scene.planes[3].d = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT    
	scene.planes[3].material.emission = vec3(0.0);
#endif
	
	scene.planes[3].material.diffuse = vec3(0.2);
	scene.planes[3].material.specular = vec3(0.1);
	scene.planes[3].material.glossiness = 100.0; 
}


void main() {
	// Setup scene
	Scene scene;
	loadScene1(scene);

	// compute color for fragment
	gl_FragColor.rgb = colorForFragment(scene, gl_FragCoord.xy);
	gl_FragColor.a = 1.0;
}
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `Tonemapping`,
		id: `CopyFS`,
		initialValue: `precision highp float;

uniform sampler2D radianceTexture;
uniform int sampleCount;
uniform ivec2 resolution;

vec3 tonemap(vec3 color, float maxLuminance, float gamma) {
	float luminance = length(color);
	//float scale =  luminance /  maxLuminance;
	float scale =  luminance / (maxLuminance * luminance + 0.0000001);
  	return max(vec3(0.0), pow(scale * color, vec3(1.0 / gamma)));
}

void main(void) {
  vec3 texel = texture2D(radianceTexture, gl_FragCoord.xy / vec2(resolution)).rgb;
  vec3 radiance = texel / float(sampleCount);
  gl_FragColor.rgb = tonemap(radiance, 1.0, 1.6);
  gl_FragColor.a = 1.0;
}
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-vertex`,
		title: ``,
		id: `VS`,
		initialValue: `
	attribute vec3 position;
	void main(void) {
		gl_Position = vec4(position, 1.0);
	}
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	 return UI; 
}//!setup


function getShader(gl, id) {

		gl.getExtension('OES_texture_float');
		//alert(gl.getSupportedExtensions());

	var shaderScript = document.getElementById(id);
	if (!shaderScript) {
		return null;
	}

	var str = "";
	var k = shaderScript.firstChild;
	while (k) {
		if (k.nodeType == 3) {
			str += k.textContent;
		}
		k = k.nextSibling;
	}

	var shader;
	if (shaderScript.type == "x-shader/x-fragment") {
		shader = gl.createShader(gl.FRAGMENT_SHADER);
	} else if (shaderScript.type == "x-shader/x-vertex") {
		shader = gl.createShader(gl.VERTEX_SHADER);
	} else {
		return null;
	}

    console.log(str);
	gl.shaderSource(shader, str);
	gl.compileShader(shader);

	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		alert(gl.getShaderInfoLog(shader));
		return null;
	}

	return shader;
}

function RaytracingDemo() {
}

function initShaders() {

	traceProgram = gl.createProgram();
	gl.attachShader(traceProgram, getShader(gl, "VS"));
	gl.attachShader(traceProgram, getShader(gl, "TraceFS"));
	gl.linkProgram(traceProgram);
	gl.useProgram(traceProgram);
	traceProgram.vertexPositionAttribute = gl.getAttribLocation(traceProgram, "position");
	gl.enableVertexAttribArray(traceProgram.vertexPositionAttribute);

	copyProgram = gl.createProgram();
	gl.attachShader(copyProgram, getShader(gl, "VS"));
	gl.attachShader(copyProgram, getShader(gl, "CopyFS"));
	gl.linkProgram(copyProgram);
	gl.useProgram(copyProgram);
	traceProgram.vertexPositionAttribute = gl.getAttribLocation(copyProgram, "position");
	gl.enableVertexAttribArray(copyProgram.vertexPositionAttribute);

}

function initBuffers() {
	triangleVertexPositionBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexPositionBuffer);

	var vertices = [
		 -1,  -1,  0,
		 -1,  1,  0,
		 1,  1,  0,

		 -1,  -1,  0,
		 1,  -1,  0,
		 1,  1,  0,
	 ];
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
	triangleVertexPositionBuffer.itemSize = 3;
	triangleVertexPositionBuffer.numItems = 3 * 2;
}


function tick() {

// 1st pass: Trace
	gl.bindFramebuffer(gl.FRAMEBUFFER, rttFramebuffer);

	gl.useProgram(traceProgram);
  	gl.uniform1i(gl.getUniformLocation(traceProgram, "globalSeed"), Math.random() * 32768.0);
	gl.uniform1i(gl.getUniformLocation(traceProgram, "baseSampleIndex"), getCurrentFrame());
	gl.uniform2i(
		gl.getUniformLocation(traceProgram, "resolution"),
		getRenderTargetWidth(),
		getRenderTargetHeight());

	gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexPositionBuffer);
	gl.vertexAttribPointer(
		traceProgram.vertexPositionAttribute,
		triangleVertexPositionBuffer.itemSize,
		gl.FLOAT,
		false,
		0,
		0);

    	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);

	gl.disable(gl.DEPTH_TEST);
	gl.enable(gl.BLEND);
	gl.blendFunc(gl.ONE, gl.ONE);

	gl.drawArrays(gl.TRIANGLES, 0, triangleVertexPositionBuffer.numItems);

// 2nd pass: Average
   	gl.bindFramebuffer(gl.FRAMEBUFFER, null);

	gl.useProgram(copyProgram);
	gl.uniform1i(gl.getUniformLocation(copyProgram, "sampleCount"), getCurrentFrame() + 1);

	gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexPositionBuffer);
	gl.vertexAttribPointer(
		copyProgram.vertexPositionAttribute,
		triangleVertexPositionBuffer.itemSize,
		gl.FLOAT,
		false,
		0,
		0);

    	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);

	gl.disable(gl.DEPTH_TEST);
	gl.disable(gl.BLEND);

	gl.activeTexture(gl.TEXTURE0);
    	gl.bindTexture(gl.TEXTURE_2D, rttTexture);
	gl.uniform1i(gl.getUniformLocation(copyProgram, "radianceTexture"), 0);
	gl.uniform2i(
		gl.getUniformLocation(copyProgram, "resolution"),
		getRenderTargetWidth(),
		getRenderTargetHeight());

	gl.drawArrays(gl.TRIANGLES, 0, triangleVertexPositionBuffer.numItems);

	gl.bindTexture(gl.TEXTURE_2D, null);
}

function init() {
	initShaders();
	initBuffers();
	gl.clear(gl.COLOR_BUFFER_BIT);

	rttFramebuffer = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, rttFramebuffer);

	rttTexture = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, rttTexture);
    	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, getRenderTargetWidth(), getRenderTargetHeight(), 0, gl.RGBA, gl.FLOAT, null);

	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, rttTexture, 0);
}

var oldWidth = 0;
var oldTraceProgram;
var oldCopyProgram;
function compute(canvas) {

	if(	getRenderTargetWidth() != oldWidth ||
		oldTraceProgram != document.getElementById("TraceFS") ||
		oldCopyProgram !=  document.getElementById("CopyFS"))
	{
		init();

		oldWidth = getRenderTargetWidth();
		oldTraceProgram = document.getElementById("TraceFS");
		oldCopyProgram = document.getElementById("CopyFS");
	}

	tick();
}
