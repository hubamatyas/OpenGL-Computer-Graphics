function setup()
{
	UI = {};
	UI.tabs = [];
	UI.titleLong = 'Ray Tracer';
	UI.titleShort = 'RayTracerSimple';
	UI.numFrames = 100000;
	UI.maxFPS = 24;
	UI.renderWidth = 1600;
	UI.renderHeight = 800;

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `RaytracingDemoFS - GL`,
		id: `RaytracingDemoFS`,
		initialValue: ` 
#define SOLUTION_CYLINDER_AND_PLANE
#define SOLUTION_SHADOW
#define SOLUTION_REFLECTION_REFRACTION
#define SOLUTION_FRESNEL

#define SOLUTION_POLYTOPE

precision highp float;
uniform ivec2 viewport; 

struct PointLight {
	vec3 position;
	vec3 color;
};

struct Material {
	vec3  diffuse;
	vec3  specular;
	float glossiness;
	float reflection;
	float refraction;
	float ior;
};

struct Sphere {
	vec3 position;
	float radius;
	Material material;
};

struct Plane {
	vec3 normal;
	float d;
	Material material;
};

struct Cylinder {
	vec3 position;
	vec3 direction;  
	float radius;
	Material material;
};


const int polytope_size = 10;
struct Polytope {
	vec3 normals[polytope_size];
	float ds[polytope_size];
	Material material;
};


const int lightCount = 2;
const int sphereCount = 3;
const int planeCount = 1;
const int cylinderCount = 2;
const int booleanCount = 2; 

struct Scene {
	vec3 ambient;
	PointLight[lightCount] lights;
	Sphere[sphereCount] spheres;
	Plane[planeCount] planes;
	Cylinder[cylinderCount] cylinders;
	Polytope polytope; 
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
	bool enteringPrimitive;
};

HitInfo getEmptyHit() {
	return HitInfo(
		false, 
		0.0, 
		vec3(0.0), 
		vec3(0.0), 
		Material(vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0),
		false);
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

// Get the smallest t in an interval.
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

HitInfo intersectSphere(const Ray ray, const Sphere sphere, const float tMin, const float tMax) {
              
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
		
		//Checking if we're inside the sphere by checking if the ray's origin is inside. If we are, then the normal 
		//at the intersection surface points towards the center. Otherwise, if we are outside the sphere, then the normal 
		//at the intersection surface points outwards from the sphere's center. This is important for refraction.
      	vec3 normal = 
          	length(ray.origin - sphere.position) < sphere.radius + 0.001? 
          	-normalize(hitPosition - sphere.position): 
      		normalize(hitPosition - sphere.position);      
		
		//Checking if we're inside the sphere by checking if the ray's origin is inside,
		// but this time for IOR bookkeeping. 
		//If we are inside, set a flag to say we're leaving. If we are outside, set the flag to say we're entering.
		//This is also important for refraction.
		bool enteringPrimitive = 
          	length(ray.origin - sphere.position) < sphere.radius + 0.001 ? 
          	false:
		    true; 

        return HitInfo(
          	true,
          	smallestTInInterval,
          	hitPosition,
          	normal,
          	sphere.material,
			enteringPrimitive);
    }
    return getEmptyHit();
}

HitInfo intersectPlane(const Ray ray,const Plane plane, const float tMin, const float tMax) {
#ifdef SOLUTION_CYLINDER_AND_PLANE
	// General formula for finding intersection of line and plane: dot(x, n) - d = 0
	// Explanation: float dist calculates the distance a ray (ie. ray.direction) that's perpendicular to the plane
	// would have to travel from ray.origin to intersect the plane. float scalar adjusts the distance given how
	// similar the ray is to the plane.normal. By definition plane.normal is perpendicular to the plane, therefore
	// the more similar (ie. greater dot product) ray.direction is to plane.normal the less it would need to travel.
	float dist = plane.d - dot(plane.normal, ray.origin);
	float scalar = dot(ray.direction, plane.normal);
	float t = dist / scalar;
	
	if (t > 0.001) {
		vec3 hitPosition = ray.origin + t * ray.direction;
		
		// Explanation: No need to account for the direction of the normal and the enteringPrimitive flag here. Where these
		// properties are needed (ie. intersectPolytope), they are calculated in a self-contained way.
		return HitInfo(
			true,
			t,
			hitPosition,
			plane.normal,
			plane.material,
			true);
	}
	
#endif  
	return getEmptyHit();
}

float lengthSquared(vec3 x) {
	return dot(x, x);
}

HitInfo intersectCylinder(const Ray ray, const Cylinder cylinder, const float tMin, const float tMax) {
#ifdef SOLUTION_CYLINDER_AND_PLANE
	// Explanation: Very similiar closed form quadratic function to the ray-sphere intersection formula.
	// Additional calculations (eg. ray_cylinder_i, x_i) account for the fact that the cylinder goes
	// on forever in the scene along its direction (ie.cylinder.direction).
	vec3 to_cylinder = ray.origin - cylinder.position;
	
	float ray_cylinder1 = dot(ray.direction, cylinder.direction);
	vec3 x1 = ray.direction - ray_cylinder1 * cylinder.direction;
	float a = dot(x1, x1);
	
	float ray_cylinder2 = dot(to_cylinder, cylinder.direction);
	vec3 x2 = to_cylinder - ray_cylinder2 * cylinder.direction;
	float b = 2.0 * dot(x1, x2);
	
	vec3 x3 = to_cylinder - ray_cylinder2 * cylinder.direction;
	float c = dot(x3, x3) - cylinder.radius * cylinder.radius;
  
    float D = b * b - 4.0 * a * c;
    if (D > 0.0)
    {
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);
      
      	float smallestTInInterval;
      	if(!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
          return getEmptyHit();
        }
		
		// Explanation: Checking if we're inside the cylinder by checking if the ray's origin is inside
		// the same way as with spheres. Set normal and enteringPrimitive based on whether we're inside
		// or outside of the cylinder.
      	vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;
      	float height = dot(hitPosition - cylinder.position, cylinder.direction);
		vec3 normal = 
			length(ray.origin - cylinder.position) < cylinder.radius + 0.001 ?
			-normalize(hitPosition - cylinder.position - height * cylinder.direction):
			normalize(hitPosition - cylinder.position - height * cylinder.direction);
				
		bool enteringPrimitive = 
          	length(ray.origin - cylinder.position) < cylinder.radius + 0.001 ? 
          	false:
		    true; 

        return HitInfo(
          	true,
          	smallestTInInterval,
          	hitPosition,
          	normal,
          	cylinder.material,
			enteringPrimitive);
    }
	
#endif  
    return getEmptyHit();
}

bool inside(const vec3 position, const Sphere sphere) {
	return length(position - sphere.position) < sphere.radius;
}

HitInfo intersectPolytope(const Ray ray, const Polytope polytope, const float tMin, const float tMax) {

#ifdef SOLUTION_POLYTOPE
	HitInfo bestHit;
	bestHit.t = tMax;
	bool isOriginInside = true;
	for(int i = 0; i < polytope_size; i++) {
		// Construct each polytope
		vec3 normal = polytope.normals[i];
		float d = polytope.ds[i];
		Plane plane = Plane(normal, d, polytope.material);
		
		HitInfo currentHit = intersectPlane(ray, plane, tMin, tMax);
		bool isHitInside = true;
		
		// Check if ray intersected with plane and whether the hit point is the closest hit point
		//  to ray.origin that has been encountered so far.
		if(currentHit.hit && currentHit.t < bestHit.t) {
			for(int j = 0; j < polytope_size; j++) {
				// Explanation: Test if currentHit.position is on the 'inside' of each plane, except
				// the plane where the intersection was calculated in the first place.
				if(j != i) {
					vec3 testNormal = polytope.normals[j];
					float testD = polytope.ds[j];
					
					// Explanation: If the dot product is greater than the distance (testD), it means the given hit point
					// (currentHit.position) is further away from the ray.origin than the plane is from the
					// ray.origin. In other words, the point is on the 'outside' of the plane. In this case
					// the hit point shouldn't be shown as it's not in the union of the planes.
					float hitDist = testD - dot(testNormal, currentHit.position);
					
					if(hitDist < 0.0) {
						isHitInside = false;
					}
				}
			}
			
			if(isHitInside) {
				bestHit = currentHit;
			}
		}
		
		// Explanation: Keep track of whether the ray origin is inside the poltyope (ie. in the union of all inner planes)
		// in the same way we check if a hit point is inside the polytope.
		float originDist = d - dot(normal, ray.origin);
		
		// 0.001 accounts for precision error
		if(originDist + 0.001 < 0.0) {
			isOriginInside = false;
		}
		
	}
	
	bestHit.normal =
		isOriginInside?
		-bestHit.normal:
		bestHit.normal;
	
	bestHit.enteringPrimitive =
		isOriginInside?
		false:
		true;
		
	return bestHit;
#else
	// Put your Polytope intersection code in the #ifdef above!
#endif
	return getEmptyHit();
}

uniform float time;

HitInfo getBetterHitInfo(const HitInfo oldHitInfo, const HitInfo newHitInfo) {
	if(newHitInfo.hit)
  		if(newHitInfo.t < oldHitInfo.t)  // No need to test for the interval, this has to be done per-primitive
          return newHitInfo;
  	return oldHitInfo;
}

HitInfo intersectScene(const Scene scene, const Ray ray, const float tMin, const float tMax) {
	HitInfo bestHitInfo;
	bestHitInfo.t = tMax;
	bestHitInfo.hit = false;
	
	bestHitInfo = getBetterHitInfo(bestHitInfo, intersectPolytope(ray, scene.polytope, tMin, tMax));

	for (int i = 0; i < planeCount; ++i) {
		bestHitInfo = getBetterHitInfo(bestHitInfo, intersectPlane(ray, scene.planes[i], tMin, tMax));
	}
	for (int i = 0; i < sphereCount; ++i) {
		bestHitInfo = getBetterHitInfo(bestHitInfo, intersectSphere(ray, scene.spheres[i], tMin, tMax));
	}
	for (int i = 0; i < cylinderCount; ++i) {
		bestHitInfo = getBetterHitInfo(bestHitInfo, intersectCylinder(ray, scene.cylinders[i], tMin, tMax));
	}
	
	return bestHitInfo;
}

vec3 shadeFromLight(
  const Scene scene,
  const Ray ray,
  const HitInfo hit_info,
  const PointLight light)
{ 
  vec3 hitToLight = light.position - hit_info.position;
  
  vec3 lightDirection = normalize(hitToLight);
  vec3 viewDirection = normalize(hit_info.position - ray.origin);
  vec3 reflectedDirection = reflect(viewDirection, hit_info.normal);
  float diffuse_term = max(0.0, dot(lightDirection, hit_info.normal));
  float specular_term  = pow(max(0.0, dot(lightDirection, reflectedDirection)), hit_info.material.glossiness);

#ifdef SOLUTION_SHADOW
  // Explanation: Check if there's anything between the initial hit point (ie. hit_info.position) and the given
  // light source (ie. PointLight light). If yes, reduce visibility of the hit point, otherwise leave it as is.
  float visibility = 1.0;
  Ray shadowRay;
  shadowRay.origin = hit_info.position;
  shadowRay.direction = lightDirection;
  HitInfo shadowHitInfo = intersectScene(scene, shadowRay, 0.001, length(hitToLight));
	
  if (shadowHitInfo.hit) {
	  visibility = 0.0;
  }
#else
  // Put your shadow test here
  float visibility = 1.0;  
#endif
  return 	visibility * 
    		light.color * (
    		specular_term * hit_info.material.specular +
      		diffuse_term * hit_info.material.diffuse);
}

vec3 background(const Ray ray) {
  // A simple implicit sky that can be used for the background
  return vec3(0.2) + vec3(0.8, 0.6, 0.5) * max(0.0, ray.direction.y);
}

// It seems to be a WebGL issue that the third parameter needs to be inout instea dof const on Tobias' machine
vec3 shade(const Scene scene, const Ray ray, inout HitInfo hitInfo) {
	
  	if(!hitInfo.hit) {
  		return background(ray);
  	}
  
    vec3 shading = scene.ambient * hitInfo.material.diffuse;
    for (int i = 0; i < lightCount; ++i) {
        shading += shadeFromLight(scene, ray, hitInfo, scene.lights[i]); 
    }
    return shading;
}


Ray getFragCoordRay(const vec2 frag_coord) {
  	float sensorDistance = 1.0;
  	vec2 sensorMin = vec2(-1, -0.5);
  	vec2 sensorMax = vec2(1, 0.5);
  	vec2 pixelSize = (sensorMax- sensorMin) / vec2(viewport.x, viewport.y);
  	vec3 origin = vec3(0, 0, sensorDistance);
    vec3 direction = normalize(vec3(sensorMin + pixelSize * frag_coord, -sensorDistance));  
  
  	return Ray(origin, direction);
}

float fresnel(const vec3 viewDirection, const vec3 normal, const float sourceIOR, const float destIOR) {
#ifdef SOLUTION_FRESNEL
	// Explanation: Schlick's approximation for Fresnel factor using the formula in https://en.wikipedia.org/wiki/Schlick%27s_approximation
	// Fresnel factor ensures the reflection and refraction weights are representative of the source and destination IORs and
	// the angle between the view direction and normal of the given object. When calling fresnel() the assumption is
	// w_reflect + w_refract <= 1, hence settting w_refract to 1.0 - fresnel() below.
    float cosTheta = dot(-viewDirection, normal);
    float r0 = (sourceIOR - destIOR) / (sourceIOR + destIOR);
    r0 = r0 * r0;
	// Exponent was chosen by trial and error. 1.5 provides the closest match to the example image in the coursework description.
    float fresnel = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 1.5);
  
    return fresnel;
#else
  	// Put your code to compute the Fresnel effect in the ifdef above
	return 1.0;
#endif
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {
      
    Ray initialRay = getFragCoordRay(fragCoord);
  	HitInfo initialHitInfo = intersectScene(scene, initialRay, 0.001, 10000.0);
  	vec3 result = shade(scene, initialRay, initialHitInfo);
	
  	Ray currentRay;
  	HitInfo currentHitInfo;
  	
  	// Compute the reflection
  	currentRay = initialRay;
  	currentHitInfo = initialHitInfo;
  	
  	// The initial strength of the reflection
  	float reflectionWeight = 1.0;
	
	// The initial medium is air
  	float currentIOR = 1.0;
	
    float sourceIOR = 1.0;
	float destIOR = 1.0;
  	
  	const int maxReflectionStepCount = 2;
  	for(int i = 0; i < maxReflectionStepCount; i++) {
      
      if(!currentHitInfo.hit) break;
      
#ifdef SOLUTION_REFLECTION_REFRACTION
	  reflectionWeight *= currentHitInfo.material.reflection;
#else
      // Put your reflection weighting code in the ifdef above
#endif
      
#ifdef SOLUTION_FRESNEL
	  reflectionWeight *= fresnel(currentRay.direction, currentHitInfo.normal, sourceIOR, destIOR);
#else
      // Replace with Fresnel code in the ifdef above
      reflectionWeight *= 0.5;
#endif
      Ray nextRay;
#ifdef SOLUTION_REFLECTION_REFRACTION
	  // Update ray to represent the hit point. Use the new ray with the reflected direction to cast reflections.
	  nextRay.origin = currentHitInfo.position;
	  nextRay.direction = reflect(normalize(currentRay.direction), currentHitInfo.normal);
	  
#else
	// Put your code to compute the reflection ray in the ifdef above
#endif
      currentRay = nextRay;
      
      currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);      
            
      result += reflectionWeight * shade(scene, currentRay, currentHitInfo);
    }
  
  	// Compute the refraction
  	currentRay = initialRay;
  	currentHitInfo = initialHitInfo;
   
  	// The initial strength of the refraction.
  	float refractionWeight = 1.0;
  
  	const int maxRefractionStepCount = 2;
  	for(int i = 0; i < maxRefractionStepCount; i++) {
      
#ifdef SOLUTION_REFLECTION_REFRACTION
	  refractionWeight *= currentHitInfo.material.refraction;
#else
      // Put your refraction weighting code in the ifdef above
      refractionWeight *= 0.5;
#endif

#ifdef SOLUTION_FRESNEL
	  // 1.0 - fresnel() because of the assumption that w_reflect + w_refract <= 1.
	  refractionWeight *= 1.0 - fresnel(currentRay.direction, currentHitInfo.normal, sourceIOR, destIOR);
#else
      // Put your Fresnel code in the ifdef above 
#endif      

	  Ray nextRay;
		
#ifdef SOLUTION_REFLECTION_REFRACTION
	  // Keep track of materials' IORs to cast the correct refractions.
	  if(currentHitInfo.enteringPrimitive) {
		  sourceIOR = currentIOR;
		  destIOR = currentHitInfo.material.ior;
	  } else {
		  sourceIOR = currentHitInfo.material.ior;
		  destIOR = currentIOR;
	  }
		
	  // Update ray to represent the hit point. Use the new ray with the refracted direction to cast refractions.
	  nextRay.origin = currentHitInfo.position;
	  nextRay.direction = refract(normalize(currentRay.direction), currentHitInfo.normal, sourceIOR/destIOR);
	  currentRay = nextRay;
#else
      float sourceIOR;
	  float destIOR;
	// Put your code to compute the reflection ray and track the IOR in the ifdef above
#endif
      currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
            
      result += refractionWeight * shade(scene, currentRay, currentHitInfo);
      
      if(!currentHitInfo.hit) break;
    }
  return result;
}

Material getDefaultMaterial() {
  return Material(vec3(0.3), vec3(0), 0.0, 0.0, 0.0, 0.0);
}

Material getPaperMaterial() {
  return Material(vec3(0.7, 0.7, 0.7), vec3(0, 0, 0), 5.0, 0.0, 0.0, 0.0);
}

Material getPlasticMaterial() {
	return Material(vec3(0.9, 0.3, 0.1), vec3(1.0), 10.0, 0.9, 0.0, 0.0);
}

Material getGlassMaterial() {
	return Material(vec3(0.0), vec3(0.0), 5.0, 1.0, 1.0, 1.5);
}

Material getSteelMirrorMaterial() {
	return Material(vec3(0.1), vec3(0.3), 20.0, 0.8, 0.0, 0.0);
}

Material getMetaMaterial() {
	return Material(vec3(0.1, 0.2, 0.5), vec3(0.3, 0.7, 0.9), 20.0, 0.8, 0.0, 0.0);
}

vec3 tonemap(const vec3 radiance) {
  const float monitorGamma = 2.0;
  return pow(radiance, vec3(1.0 / monitorGamma));
}

void clearShape(inout Polytope shape) {
	/*
		clear the polytope
	*/
	for (int i = 0; i < polytope_size; i++) {
		shape.normals[i] = vec3(0.0);
		shape.ds[i] = 0.0;
	}
}
void loadCube(float size, inout Polytope cube) {
	/* 
		load a cube to test intersection code
		
		NOTE THAT:
		Here the cube is loaded in a specific order => (top, bottom, front, back, left, right).
		You should load the diamond in a similar way.
	*/
	cube.normals[0] = (vec3(0, 1, 0)); // TOP
	cube.normals[1] = vec3(0, -1, 0); // BOTTOM
	cube.normals[2] = vec3(0, 0, 1); // FRONT
	cube.normals[3] = vec3(0, 0, -1); // BACK
	cube.normals[4] = vec3(-1, 0, 0); // LEFT
	cube.normals[5] = vec3(1, 0, 0); // RIGHT
	for (int i = 0; i < 6; i++) {
		cube.ds[i] = size; // the size of the cube
	}
}

void loadDiamond(float height, float width, float slope, inout Polytope diamond, inout bool isSuccessful) {
	/*
		Implement your code to load a diamond at origin (0.0, 0.0, 0.0)
		
		Input Arguments:
			height: the distance between the top plane and the bottom plane
			width: the edge length of the square in the middle of the diamond
			slope: the angle between the 8 side planes and the X-Z plane (in radians)
			
		Inout Arguments:
			diamond: the polytope that you need to modify
			isSuccessful: true if diamond is loaded successfully, false otherwise
		
		NOTE THAT:
		Please load the diamond, i.e., 10 planes, in this order:
		(top, bottom, upper-front, upper-back, upper-left, upper-right, lower-front, lower-back, lower-left, lower-right)
	*/

	float halfHeight = height / 2.0;
    float halfWidth = width / 2.0;
    float cosSlope = cos(slope);
    float sinSlope = sin(slope);
	float dist = halfWidth * sinSlope;

    diamond.normals[0] = vec3(0.0, 1.0, 0.0); // Top
    diamond.ds[0] = halfHeight;

    diamond.normals[1] = vec3(0.0, -1.0, 0.0); // Bottom
    diamond.ds[1] = halfHeight;

	// Explanation: Tilt each plane using sin(slope) and cos(slope) to get the desired union
	// Distance is constant as the diamond is symmetric.
    diamond.normals[2] = vec3(0.0, cosSlope, sinSlope); // Upper Front
    diamond.ds[2] = dist;

    diamond.normals[3] = vec3(0.0, cosSlope, -sinSlope); // Upper Back
    diamond.ds[3] = dist;

    diamond.normals[4] = vec3(-sinSlope, cosSlope, 0.0); // Upper Left
    diamond.ds[4] = dist;

    diamond.normals[5] = vec3(sinSlope, cosSlope, 0.0); // Upper Right
    diamond.ds[5] = dist;

    diamond.normals[6] = vec3(0.0, -cosSlope, sinSlope); // Lower Front
    diamond.ds[6] = dist;

    diamond.normals[7] = vec3(0.0, -cosSlope, -sinSlope); // Lower Back
    diamond.ds[7] = dist;

    diamond.normals[8] = vec3(-sinSlope, -cosSlope, 0.0); // Lower Left
    diamond.ds[8] = dist;

    diamond.normals[9] = vec3(sinSlope, -cosSlope, -0.0); // Lower Right
    diamond.ds[9] = dist;

    isSuccessful = true;
	
#ifdef SOLUTION_POLYTOPE
#else
	// Put your code in the above solution block
	// don't forget to set it true in your implementation!
	isSuccessful = false;
#endif
}

void main() {
	// Setup scene
	Scene scene;
	scene.ambient = vec3(0.12, 0.15, 0.2);
	scene.lights[0].position = vec3(5, 15, -5);
	scene.lights[0].color    = 0.5 * vec3(0.9, 0.5, 0.1);

	scene.lights[1].position = vec3(-15, 5, 2);
	scene.lights[1].color    = 0.5 * vec3(0.1, 0.3, 1.0);
	
	// Primitives
	bool specialScene = false;
	
#if defined(SOLUTION_POLYTOPE)
	specialScene = true;
#endif
	
	if (specialScene) {
		// Polytope diamond scene
		float slope = radians(70.0);
		bool isDiamond = false;
		loadDiamond(8.0, 4.37, slope, scene.polytope, isDiamond);
		
		if (!isDiamond) {
			clearShape(scene.polytope);
			loadCube(3.0, scene.polytope);
		}
		
		// rotate the diamond along the Y axis
		mat3 rot;
		float speed = 10.0;
		float theta = radians(speed * time);
		
		// Three angles that might be tested for marking
		// theta = radians(0.0);
		// theta = radians(30.0);
		// theta = radians(45.0);
		
		// rotating
		rot[0] = vec3(cos(theta), 0, -sin(theta));
		rot[1] = vec3(0, 1, 0);
		rot[2] = vec3(sin(theta), 0, cos(theta));
		for (int i = 0; i < polytope_size; i++) {
			scene.polytope.normals[i] = rot * scene.polytope.normals[i];
		}
		
		// push the origin-centered diamond along the Z axis to display it properly
		float at_z = 15.0;
		// push the upper part
		scene.polytope.ds[2] = scene.polytope.ds[2] - cos(theta) * sin(slope) * at_z;
		scene.polytope.ds[3] = scene.polytope.ds[3] + cos(theta) * sin(slope) * at_z;
		scene.polytope.ds[4] = scene.polytope.ds[4] - sin(theta) * sin(slope) * at_z;
		scene.polytope.ds[5] = scene.polytope.ds[5] + sin(theta) * sin(slope) * at_z;
		if (isDiamond) {
			// push the lower part
			scene.polytope.ds[6] = scene.polytope.ds[6] - cos(theta) * sin(slope) * at_z;
			scene.polytope.ds[7] = scene.polytope.ds[7] + cos(theta) * sin(slope) * at_z;
			scene.polytope.ds[8] = scene.polytope.ds[8] - sin(theta) * sin(slope) * at_z;
			scene.polytope.ds[9] = scene.polytope.ds[9] + sin(theta) * sin(slope) * at_z;
		}
		
		if (isDiamond) scene.polytope.material = getGlassMaterial();
		else scene.polytope.material = getMetaMaterial();
		
		// add floor
		scene.planes[0].normal            		= normalize(vec3(0, 1.0, 0));
		scene.planes[0].d              			= -4.5;
		scene.planes[0].material				= getSteelMirrorMaterial();
		if (isDiamond) {
			// add some primitives to play around
			scene.cylinders[0].position            	= vec3(-15, 1, -26);
			scene.cylinders[0].direction            = normalize(vec3(-2, 2, -1));
			scene.cylinders[0].radius         		= 1.5;
			scene.cylinders[0].material				= getPaperMaterial();

			scene.cylinders[1].position            	= vec3(15, 1, -26);
			scene.cylinders[1].direction            = normalize(vec3(2, 2, -1));
			scene.cylinders[1].radius         		= 1.5;
			scene.cylinders[1].material				= getPlasticMaterial();
		}
	}
	else {
		// normal scene
		scene.spheres[0].position            	= vec3(10, -5, -16);
		scene.spheres[0].radius              	= 6.0;
		scene.spheres[0].material 				= getPaperMaterial();

		scene.spheres[1].position            	= vec3(-7, -2, -13);
		scene.spheres[1].radius             	= 4.0;
		scene.spheres[1].material				= getPlasticMaterial();

		scene.spheres[2].position            	= vec3(0, 0.5, -5);
		scene.spheres[2].radius              	= 2.0;
		scene.spheres[2].material   			= getGlassMaterial();

		scene.planes[0].normal            		= normalize(vec3(0.0, 1.0, 0.0));
		scene.planes[0].d              			= -4.5;
		scene.planes[0].material				= getSteelMirrorMaterial();

		scene.cylinders[0].position            	= vec3(-1, 1, -26);
		scene.cylinders[0].direction            = normalize(vec3(-2, 2, -1));
		scene.cylinders[0].radius         		= 1.5;
		scene.cylinders[0].material				= getPaperMaterial();

		scene.cylinders[1].position            	= vec3(4, 1, -5);
		scene.cylinders[1].direction            = normalize(vec3(1, 4, 1));
		scene.cylinders[1].radius         		= 0.4;
		scene.cylinders[1].material				= getPlasticMaterial();
	}

	// Compute color for fragment
	gl_FragColor.rgb = tonemap(colorForFragment(scene, gl_FragCoord.xy));
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
		title: `RaytracingDemoVS - GL`,
		id: `RaytracingDemoVS`,
		initialValue: `attribute vec3 position;
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
  
    void main(void) {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	 return UI; 
}//!setup

var gl;
function initGL(canvas) {
	try {
		gl = canvas.getContext("experimental-webgl");
		gl.viewportWidth = canvas.width;
		gl.viewportHeight = canvas.height;
	} catch (e) {
	}
	if (!gl) {
		alert("Could not initialise WebGL, sorry :-(");
	}
}

function getShader(gl, id) {
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

RaytracingDemo.prototype.initShaders = function() {

	this.shaderProgram = gl.createProgram();

	gl.attachShader(this.shaderProgram, getShader(gl, "RaytracingDemoVS"));
	gl.attachShader(this.shaderProgram, getShader(gl, "RaytracingDemoFS"));
	gl.linkProgram(this.shaderProgram);

	if (!gl.getProgramParameter(this.shaderProgram, gl.LINK_STATUS)) {
		alert("Could not initialise shaders");
	}

	gl.useProgram(this.shaderProgram);

	this.shaderProgram.vertexPositionAttribute = gl.getAttribLocation(this.shaderProgram, "position");
	gl.enableVertexAttribArray(this.shaderProgram.vertexPositionAttribute);

	this.shaderProgram.projectionMatrixUniform = gl.getUniformLocation(this.shaderProgram, "projectionMatrix");
	this.shaderProgram.modelviewMatrixUniform = gl.getUniformLocation(this.shaderProgram, "modelViewMatrix");
}

RaytracingDemo.prototype.initBuffers = function() {
	this.triangleVertexPositionBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
	
	var vertices = [
		 -1,  -1,  0,
		 -1,  1,  0,
		 1,  1,  0,

		 -1,  -1,  0,
		 1,  -1,  0,
		 1,  1,  0,
	 ];
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
	this.triangleVertexPositionBuffer.itemSize = 3;
	this.triangleVertexPositionBuffer.numItems = 3 * 2;
}

function getTime() {  
	var d = new Date();
	return d.getMinutes() * 60.0 + d.getSeconds() + d.getMilliseconds() / 1000.0;
}

RaytracingDemo.prototype.drawScene = function() {
			
	var perspectiveMatrix = new J3DIMatrix4();	
	perspectiveMatrix.setUniform(gl, this.shaderProgram.projectionMatrixUniform, false);

	var modelViewMatrix = new J3DIMatrix4();	
	modelViewMatrix.setUniform(gl, this.shaderProgram.modelviewMatrixUniform, false);

	gl.uniform1f(gl.getUniformLocation(this.shaderProgram, "time"), getTime());
	
	gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
	gl.vertexAttribPointer(this.shaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
	
	gl.uniform2iv(gl.getUniformLocation(this.shaderProgram, "viewport"), [getRenderTargetWidth(), getRenderTargetHeight()]);

	gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);
}

RaytracingDemo.prototype.run = function() {
	this.initShaders();
	this.initBuffers();

	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
	gl.clear(gl.COLOR_BUFFER_BIT);

	this.drawScene();
};

function init() {	
	

	env = new RaytracingDemo();	
	env.run();

    return env;
}

function compute(canvas)
{
    env.initShaders();
    env.initBuffers();

    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT);

    env.drawScene();
}
