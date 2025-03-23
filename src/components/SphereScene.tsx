import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { vectors, Vector6D } from "../data/spheres";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { getClusterAnalysis } from "../api/apiClient";
import { TextGeometry } from "three/addons/geometries/TextGeometry.js";
import { FontLoader } from "three/addons/loaders/FontLoader.js";

// Extended interface for our data points
interface DataPoint extends Vector6D {
  size?: number; // Size factor (1.0 is default)
  cluster?: number; // Cluster ID
  confidence?: number; // Confidence score (0-1)
  connections?: number[]; // Indices of connected data points
}

// Define cluster colors
const clusterColors = [
  new THREE.Color(0x4285f4), // Blue
  new THREE.Color(0xea4335), // Red
  new THREE.Color(0xfbbc05), // Yellow
  new THREE.Color(0x34a853), // Green
  new THREE.Color(0x8f00ff), // Purple
  new THREE.Color(0xff6d01), // Orange
  new THREE.Color(0x00ffff), // Cyan
  new THREE.Color(0xff00ff), // Magenta
  new THREE.Color(0xc71585), // Medium Violet Red
  new THREE.Color(0x20b2aa), // Light Sea Green
];

// Define interface for embeddings data
interface EmbeddingsData {
  dimensions: number;
  count: number;
  embeddings: number[][];
  file_size_bytes: number;
}

// Add this interface definition for the cluster data structure
interface ClusterEmbedding {
  id: number;
  resume_id: number;
  cluster_id: number;
  embedding: number[];
}

// Update the ClusterData interface to match actual data structure
interface ClusterData {
  cluster_id?: number;
  total_embeddings?: number;
  dimensions?: number;
  embeddings?: ClusterEmbedding[];
  clusters?: {
    [clusterId: string]: {
      size: number;
      center: number[];
      embeddings?: any[];
      [key: string]: any;
    };
  };
}

// Add this interface near the top of your file
interface ClusterAnalysis {
  description: string;
  [key: string]: any;
}

// Fix duplicate interfaces and component declarations
interface SphereSceneProps {
  unbiasedEmbeddings?: EmbeddingsData | null;
  removedEmbeddings?: EmbeddingsData | null;
  activeTab?: string; // Accept any string
  clusterData?: any;
  clusterEmbeddings?: ClusterData | null; // Add with proper typing
  showDefaultObjects?: boolean; // Add this prop
  clusterCount?: number;
}

// Add this before your SphereScene component, outside any function
declare global {
  interface Window {
    apiDataPoints?: DataPoint[];
    recenterCamera?: () => void;
  }
}

export default function SphereScene({
  unbiasedEmbeddings,
  removedEmbeddings,
  activeTab = "clusters",
  clusterEmbeddings,
  clusterCount,
  showDefaultObjects = true, // Default to showing the objects
}: SphereSceneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDraggingRef = useRef(false);
  const rotationRef = useRef({ x: 0, y: 0 });
  const previousMousePositionRef = useRef({ x: 0, y: 0 });
  const [hoveredVector, setHoveredVector] = useState<DataPoint | null>(null);
  const [selectedVector, setSelectedVector] = useState<DataPoint | null>(null);
  const didMoveRef = useRef(false);
  const hoveredSphereRef = useRef<THREE.Mesh | null>(null);
  const selectedSphereRef = useRef<THREE.Mesh | null>(null);
  const cameraPositionRef = useRef({ z: 12 });
  const [activeCluster, setActiveCluster] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const pointsRef = useRef<THREE.Group | null>(null);
  const removedPointsRef = useRef<THREE.Group | null>(null);
  const defaultObjectsRef = useRef<THREE.Group | null>(null); // Add this ref

  // Add velocity tracking for momentum
  const velocityRef = useRef({ x: 0, y: 0 });
  const lastTimeRef = useRef(0);
  const momentumActiveRef = useRef(false);

  // Example: track data from embeddings if needed
  const [apiDataPoints, setApiDataPoints] = useState<any[]>([]);

  // Add this state to track visible clusters
  const [visibleClusters, setVisibleClusters] = useState<{
    [id: string]: boolean;
  }>({});

  // Add this state to store cluster analyses
  const [clusterAnalyses, setClusterAnalyses] = useState<{
    [id: string]: string;
  }>({});
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);

  // Add this state to track hovered node
  const [hoveredNode, setHoveredNode] = useState<{
    id: number;
    cluster_id: number;
    resume_id?: number;
  } | null>(null);

  // Add these new state variables to track the hover position
  const [hoverPosition, setHoverPosition] = useState<{
    x: number;
    y: number;
  } | null>(null);

  // Standalone useEffect for initializing visibility state
  useEffect(() => {
    if (!clusterEmbeddings?.clusters) return;

    // Initialize all clusters to visible
    const initialVisibility: { [id: string]: boolean } = {};
    Object.keys(clusterEmbeddings.clusters).forEach((id) => {
      initialVisibility[id] = true;
    });

    setVisibleClusters(initialVisibility);
  }, [clusterEmbeddings]);

  // Function to handle recenter button click - moved outside useEffect
  const handleRecenter = () => {
    if (window.recenterCamera) {
      window.recenterCamera();
    }
  };

  // Function to get cluster name - moved outside useEffect
  const getClusterName = (clusterId: number | undefined) => {
    if (clusterId === undefined) return "Unknown";
    const clusterNames = [
      "Blue Group",
      "Red Group",
      "Yellow Group",
      "Green Group",
    ];
    return clusterNames[clusterId % clusterNames.length];
  };

  // 1. First effect: basic scene setup (if clusters tab)
  useEffect(() => {
    if (activeTab !== "clusters" || !canvasRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8f9fa);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 6;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;

    // Create a group to hold all spheres
    const sphereGroup = new THREE.Group();
    scene.add(sphereGroup);

    // Create a group for connection lines
    const lineGroup = new THREE.Group();
    sphereGroup.add(lineGroup);

    // Create default objects (red sphere and text)
    const defaultGroup = new THREE.Group();
    scene.add(defaultGroup);
    defaultObjectsRef.current = defaultGroup;

    // Add a red sphere
    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshStandardMaterial({
      color: 0xff3333,
      roughness: 0.3,
      metalness: 0.7,
    });
    const sphere = new THREE.Mesh(geometry, material);
    defaultGroup.add(sphere);

    // Add text using a canvas texture
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    if (context) {
      canvas.width = 512;
      canvas.height = 100;
      context.fillStyle = "black";
      context.font = "bold 50px Arial";
      context.textAlign = "center";

      const texture = new THREE.CanvasTexture(canvas);
      const textPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(4, 1),
        new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          side: THREE.DoubleSide,
        })
      );
      textPlane.position.y = 2;
      defaultGroup.add(textPlane);
    }

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    // Regular render loop
    const animate = () => {
      requestAnimationFrame(animate);

      // Rotate the default sphere if it exists
      if (defaultObjectsRef.current && defaultObjectsRef.current.visible) {
        sphere.rotation.y += 0.01;
        sphere.rotation.x += 0.005;
      }

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      renderer.dispose();
      controls.dispose();
    };
  }, [activeTab]);

  // Add effect to show/hide default objects based on prop
  useEffect(() => {
    if (defaultObjectsRef.current) {
      defaultObjectsRef.current.visible = showDefaultObjects;
    }
  }, [showDefaultObjects]);

  // 2. Second effect: convert unbiasedEmbeddings into data points if clusters tab
  useEffect(() => {
    if (activeTab !== "clusters" || !unbiasedEmbeddings?.embeddings) return;

    try {
      const points = unbiasedEmbeddings.embeddings
        .map((emb) => {
          // Make sure there's enough length
          if (emb.length < 6) return null;
          // Example: convert to a data point
          return {
            x: emb[0],
            y: emb[1],
            z: emb[2],
            r: (emb[3] + 1) / 2,
            g: (emb[4] + 1) / 2,
            b: (emb[5] + 1) / 2,
          };
        })
        .filter(Boolean);

      setApiDataPoints(points as any[]);
    } catch (error) {
      console.error("Error converting embeddings:", error);
      setApiDataPoints([]);
    }
  }, [unbiasedEmbeddings, activeTab]);

  // 3. Third effect: actually create spheres (or points) from dataPoints if clusters tab
  useEffect(() => {
    if (activeTab !== "clusters") return;
    if (!sceneRef.current) return;

    // Clear previous objects if needed
    // ...

    // Create new group
    const group = new THREE.Group();

    // Example scale factor
    const SCALE = 2.5;
    // Create spheres
    apiDataPoints.forEach(({ x, y, z, r, g, b }) => {
      // Use a consistent, very small size (0.05)
      const geometry = new THREE.SphereGeometry(0.05, 12, 12);
      const material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(r, g, b),
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(x * SCALE, y * SCALE, z * SCALE);
      group.add(mesh);
    });

    sceneRef.current.add(group);

    // Cleanup
    return () => {
      sceneRef.current?.remove(group);
      group.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          if (obj.material instanceof THREE.Material) obj.material.dispose();
        }
      });
    };
  }, [apiDataPoints, activeTab]);

  // 4. Fourth effect: handle removedEmbeddings similarly if needed
  useEffect(() => {
    if (
      activeTab !== "clusters" ||
      !removedEmbeddings?.embeddings ||
      !sceneRef.current
    )
      return;

    // Example: create a group
    const removedGroup = new THREE.Group();
    const SCALE = 2.5;

    removedEmbeddings.embeddings.forEach((emb) => {
      if (emb.length < 6) return;
      // Same size for consistency
      const geometry = new THREE.SphereGeometry(0.05, 12, 12);
      const material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(
          (emb[3] + 1) / 2,
          (emb[4] + 1) / 2,
          (emb[5] + 1) / 2
        ),
        opacity: 0.7,
        transparent: true,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(emb[0] * SCALE, emb[1] * SCALE, emb[2] * SCALE);
      removedGroup.add(mesh);
    });

    sceneRef.current.add(removedGroup);

    return () => {
      sceneRef.current?.remove(removedGroup);
      removedGroup.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          if (obj.material instanceof THREE.Material) obj.material.dispose();
        }
      });
    };
  }, [removedEmbeddings, activeTab]);

  // The cluster visualization effect
  useEffect(() => {
    if (activeTab !== "clusters" || !clusterEmbeddings || !sceneRef.current) {
      return;
    }

    if (!clusterEmbeddings.clusters) {
      console.warn("No clusters found in clusterEmbeddings data");
      return;
    }

    const clusters = clusterEmbeddings.clusters;

    // Create a parent group for all clusters
    const allClustersGroup = new THREE.Group();
    const allLinesGroup = new THREE.Group();
    const SCALE = 2.5;

    // Create a fixed color map based on cluster IDs
    const clusterIds = Object.keys(clusters).map((id) => parseInt(id, 10));
    console.log(`Found ${clusterIds.length} cluster IDs:`, clusterIds);

    // Track hovered and active nodes
    let hoveredNode: THREE.Mesh | null = null;
    let selectedNode: THREE.Mesh | null = null;

    // Raycaster for interaction
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    // Clock for animation timing
    const clock = new THREE.Clock();
    clock.start();

    // Process each cluster separately
    Object.entries(clusters).forEach(
      ([clusterIdStr, clusterInfo]: [string, any], clusterIndex) => {
        const clusterId = parseInt(clusterIdStr.slice(-1), 10);

        // Skip if this cluster should be hidden
        if (visibleClusters[clusterIdStr] === false) {
          return;
        }

        const clusterColor = clusterColors[clusterIndex % clusterColors.length];
        console.log(
          `Processing cluster ${clusterId} with color #${clusterColor.getHexString()}`
        );

        if (!clusterInfo.embeddings || !Array.isArray(clusterInfo.embeddings)) {
          console.warn(`No embeddings found for cluster ${clusterId}`);
          return;
        }

        console.log(
          `Cluster ${clusterId} has ${clusterInfo.embeddings.length} embeddings`
        );

        // Create a group for this specific cluster
        const clusterGroup = new THREE.Group();
        const clusterLineGroup = new THREE.Group();

        // Keep track of nodes in this cluster for connections
        const clusterNodes: THREE.Mesh[] = [];

        // Create spheres for this cluster
        clusterInfo.embeddings.forEach((emb: any, nodeIndex: number) => {
          const embedding = emb.embedding || emb;

          // Only continue if we have enough dimensions
          if (!Array.isArray(embedding) || embedding.length < 3) {
            return;
          }

          // Create a sphere for this node - make it slightly larger (0.08 instead of 0.05)
          const geometry = new THREE.SphereGeometry(0.08, 16, 16);

          // Create a shader material for better appearance and hover effects
          const material = new THREE.ShaderMaterial({
            uniforms: {
              baseColor: { value: clusterColor },
              isHovered: { value: 0.0 },
              time: { value: 0.0 },
            },
            vertexShader: `
              varying vec3 vNormal;
              varying vec3 vPosition;
              uniform float isHovered;
              
              void main() {
                vNormal = normalize(normalMatrix * normal);
                vPosition = position;
                
                // Apply a size increase when hovered
                vec3 newPosition = position * (1.0 + isHovered * 0.3);
                gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
              }
            `,
            fragmentShader: `
              varying vec3 vNormal;
              varying vec3 vPosition;
              uniform vec3 baseColor;
              uniform float isHovered;
              uniform float time;
              
              void main() {
                // Calculate view direction
                vec3 viewDirection = normalize(cameraPosition - vPosition);
                float viewAngle = dot(vNormal, viewDirection);
                
                // Create shiny metallic effect with consistent lighting
                vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
                float diffuse = max(0.0, dot(vNormal, lightDir));
                
                // Add specular highlight with angle-dependent intensity
                vec3 halfVector = normalize(lightDir + viewDirection);
                float specular = pow(max(0.0, dot(vNormal, halfVector)), 30.0);
                
                // Scale specular based on viewing angle for more consistent appearance
                specular *= mix(0.4, 0.8, viewAngle);
                
                // Create animated shimmer effect with reduced intensity at grazing angles
                float shimmer = sin(vPosition.x * 20.0 + vPosition.y * 20.0 + vPosition.z * 20.0 + time * 3.0) * 0.5 + 0.5;
                shimmer = pow(shimmer, 4.0) * 0.3 * viewAngle;
                
                // Combine all lighting effects with increased base illumination (0.5 instead of 0.4)
                vec3 finalColor = baseColor * (0.5 + diffuse * 0.6);
                
                // Add enhanced specular highlight (0.6 instead of 0.5)
                finalColor += vec3(1.0, 1.0, 1.0) * specular * 0.6;
                
                // Add shimmer with slightly increased intensity
                finalColor += vec3(1.0, 1.0, 1.0) * shimmer * 1.2;
                
                // Add glow effect when hovered with angle-dependent intensity
                if (isHovered > 0.0) {
                  // Increase brightness and add pulsing glow
                  float pulse = sin(time * 5.0) * 0.5 + 0.5;
                  // Scale brightness increase based on viewing angle
                  float brightnessScale = mix(0.8, 1.3, viewAngle);
                  finalColor = finalColor * brightnessScale + vec3(1.0, 1.0, 1.0) * pulse * 0.3 * viewAngle;
                }
                
                gl_FragColor = vec4(finalColor, 1.0);
              }
            `,
          });

          const mesh = new THREE.Mesh(geometry, material);
          mesh.position.set(
            embedding[0] * SCALE,
            embedding[1] * SCALE,
            embedding[2] * SCALE
          );

          // Add custom properties for hover/selection
          mesh.userData = {
            id: emb.id || nodeIndex,
            cluster_id: clusterId,
            resume_id: emb.resume_id || 0,
            material: material, // Store reference to material for hover effects
          };

          clusterGroup.add(mesh);
          clusterNodes.push(mesh);
        });

        // Create connections between nodes in this cluster
        const MAX_CONNECTIONS_PER_NODE = 3;

        // For each node, connect to its closest neighbors within the same cluster
        clusterNodes.forEach((sourceNode, i) => {
          // Calculate distances to all other nodes in this cluster
          const distances = clusterNodes
            .map((targetNode, j) => {
              if (i === j) return { index: j, distance: Infinity }; // Skip self

              // Calculate Euclidean distance between nodes
              const dx = sourceNode.position.x - targetNode.position.x;
              const dy = sourceNode.position.y - targetNode.position.y;
              const dz = sourceNode.position.z - targetNode.position.z;
              const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

              return { index: j, distance };
            })
            // Sort by distance (ascending)
            .sort((a, b) => a.distance - b.distance)
            // Take only the closest few
            .slice(0, MAX_CONNECTIONS_PER_NODE);

          // Create lines to the closest neighbors
          distances.forEach(({ index: j, distance }) => {
            // Skip if distance is too large
            if (distance > 2) return;

            const points = [
              sourceNode.position.clone(),
              clusterNodes[j].position.clone(),
            ];

            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({
              color: clusterColor,
              transparent: true,
              opacity: 0.8,
              linewidth: 3,
            });

            const line = new THREE.Line(geometry, material);
            clusterLineGroup.add(line);
          });
        });

        // Add this cluster's groups to the parent groups
        allClustersGroup.add(clusterGroup);
        allLinesGroup.add(clusterLineGroup);
      }
    );

    // Add both parent groups to the scene
    sceneRef.current.add(allClustersGroup);
    sceneRef.current.add(allLinesGroup);

    // Animation loop to update shader uniforms
    const animate = () => {
      const elapsedTime = clock.getElapsedTime();

      // Update all materials
      allClustersGroup.traverse((object) => {
        if (object instanceof THREE.Mesh && object.userData.material) {
          const material = object.userData.material as THREE.ShaderMaterial;
          if (material.uniforms.time) {
            material.uniforms.time.value = elapsedTime;
          }
        }
      });

      requestAnimationFrame(animate);
    };

    animate();

    // Add mouse move handler for hover detection
    const handleMouseMove = (event: MouseEvent) => {
      // Calculate mouse position in normalized device coordinates
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      // Update the raycaster
      if (!cameraRef.current) return;
      raycaster.setFromCamera(mouse, cameraRef.current);

      // Find intersections with spheres
      const intersects = raycaster.intersectObjects(
        allClustersGroup.children,
        true
      );

      // Reset previous hover state
      if (hoveredNode && hoveredNode.userData.material) {
        (
          hoveredNode.userData.material as THREE.ShaderMaterial
        ).uniforms.isHovered.value = 0.0;
      }

      if (intersects.length > 0) {
        // Get the first intersected object
        const object = intersects[0].object as THREE.Mesh;

        // Check if it has userData
        if (object.userData && object.userData.id !== undefined) {
          // Set the hovered node data for info display
          setHoveredNode({
            id: object.userData.id,
            cluster_id: object.userData.cluster_id,
            resume_id: object.userData.resume_id,
          });

          // Calculate the position for the hover panel
          // Convert 3D position to screen coordinates
          const vector = new THREE.Vector3();
          vector.setFromMatrixPosition(object.matrixWorld);
          vector.project(cameraRef.current);

          const x = (vector.x * 0.5 + 0.5) * rect.width + rect.left;
          const y = (-(vector.y * 0.5) + 0.5) * rect.height + rect.top;

          // Set hover position (offset slightly above and to the right)
          setHoverPosition({ x: x + 20, y: y - 60 });

          // Apply hover effect
          if (object.userData.material) {
            (
              object.userData.material as THREE.ShaderMaterial
            ).uniforms.isHovered.value = 1.0;
            hoveredNode = object;
          }
        }
      } else {
        // Clear hovered node when not hovering over any sphere
        setHoveredNode(null);
        setHoverPosition(null);
        hoveredNode = null;
      }
    };

    // Add mouse leave handler
    const handleMouseLeave = () => {
      setHoveredNode(null);
      setHoverPosition(null);
      if (hoveredNode && hoveredNode.userData.material) {
        (
          hoveredNode.userData.material as THREE.ShaderMaterial
        ).uniforms.isHovered.value = 0.0;
      }
      hoveredNode = null;
    };

    // Add event listeners
    canvasRef.current?.addEventListener("mousemove", handleMouseMove);
    canvasRef.current?.addEventListener("mouseleave", handleMouseLeave);

    // Add the recenterCamera function to the window object
    window.recenterCamera = () => {
      if (!cameraRef.current || !controlsRef.current) return;

      const camera = cameraRef.current;
      const controls = controlsRef.current;

      // 1. Capture current camera position and rotation
      const startPosition = camera.position.clone();
      const startTarget = controls.target.clone();

      // 2. Define target values (centered view)
      const targetPosition = new THREE.Vector3(0, 0, 6);
      const targetTarget = new THREE.Vector3(0, 0, 0);
      const duration = 1000; // 1 second

      // 3. Set up animation
      const startTime = performance.now();
      const animateReset = (currentTime: number) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeProgress = 1 - Math.pow(1 - progress, 3); // easeOutCubic

        // Interpolate position
        camera.position.x =
          startPosition.x + (targetPosition.x - startPosition.x) * easeProgress;
        camera.position.y =
          startPosition.y + (targetPosition.y - startPosition.y) * easeProgress;
        camera.position.z =
          startPosition.z + (targetPosition.z - startPosition.z) * easeProgress;

        // Interpolate target (what the camera is looking at)
        controls.target.x =
          startTarget.x + (targetTarget.x - startTarget.x) * easeProgress;
        controls.target.y =
          startTarget.y + (targetTarget.y - startTarget.y) * easeProgress;
        controls.target.z =
          startTarget.z + (targetTarget.z - startTarget.z) * easeProgress;

        // Update controls
        controls.update();

        if (progress < 1) {
          requestAnimationFrame(animateReset);
        }
      };

      requestAnimationFrame(animateReset);
    };

    // Cleanup function
    return () => {
      console.log("Cleaning up cluster visualization");
      sceneRef.current?.remove(allClustersGroup);
      sceneRef.current?.remove(allLinesGroup);

      // Dispose of all geometries and materials
      allClustersGroup.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          if (obj.material instanceof THREE.Material) obj.material.dispose();
        }
      });

      allLinesGroup.traverse((obj) => {
        if (obj instanceof THREE.Line) {
          obj.geometry.dispose();
          if (obj.material instanceof THREE.Material) obj.material.dispose();
        }
      });

      // Remove event listeners
      canvasRef.current?.removeEventListener("mousemove", handleMouseMove);
      canvasRef.current?.removeEventListener("mouseleave", handleMouseLeave);

      // Clean up the window object
      delete window.recenterCamera;
    };
  }, [clusterEmbeddings, activeTab, visibleClusters, selectedCluster]);

  // Toggle cluster visibility handler
  const toggleClusterVisibility = (clusterId: string) => {
    setVisibleClusters((prev) => ({
      ...prev,
      [clusterId]: !prev[clusterId],
    }));
  };

  // Add this useEffect to fetch cluster analyses
  useEffect(() => {
    if (!clusterEmbeddings?.clusters) return;

    const fetchClusterAnalyses = async () => {
      const analyses: { [id: string]: string } = {};

      for (const clusterId of Object.keys(clusterEmbeddings.clusters || {})) {
        try {
          console.log(`Fetching analysis for cluster ${clusterId}`);
          const analysis = await getClusterAnalysis(
            parseInt(clusterId.substr(clusterId.length - 1), 10)
          );
          if (analysis && (analysis as any).analysis) {
            analyses[clusterId] = (analysis as any).analysis;
          } else {
            analyses[clusterId] = "No analysis available for this cluster.";
          }
        } catch (error) {
          console.error(
            `Error fetching analysis for cluster ${clusterId}:`,
            error
          );
          analyses[clusterId] = "Failed to load cluster analysis.";
        }
      }

      setClusterAnalyses(analyses);
    };

    fetchClusterAnalyses();
  }, [clusterEmbeddings]);

  // Add this handler to select a cluster for viewing its analysis
  const selectCluster = (clusterId: string) => {
    setSelectedCluster(selectedCluster === clusterId ? null : clusterId);
  };

  // Add this function to get the letter for a cluster ID
  const getClusterLetter = (clusterId: string, clusterEmbeddings: any) => {
    if (!clusterEmbeddings?.clusters) return "?";
    const clusterIds = Object.keys(clusterEmbeddings.clusters);
    const index = clusterIds.indexOf(clusterId);
    return index >= 0 ? String.fromCharCode(65 + (index % 26)) : "?";
  };

  // Update this function to clean all markdown from analysis text
  const cleanAnalysisText = (text: string) => {
    if (!text) return "";
    
    // Remove all markdown formatting
    return text
      // Remove bold formatting
      .replace(/\*\*(.*?)\*\*/g, "$1")
      // Remove italic formatting
      .replace(/\*(.*?)\*/g, "$1")
      // Remove heading markers
      .replace(/^#+\s+/gm, "")
      // Remove backticks for code
      .replace(/`(.*?)`/g, "$1")
      // Remove links but keep the text
      .replace(/\[(.*?)\]\(.*?\)/g, "$1")
      // Remove horizontal rules
      .replace(/^\s*[-*_]{3,}\s*$/gm, "")
      // Remove blockquotes
      .replace(/^>\s+/gm, "")
      // Remove list markers
      .replace(/^[\s-]*[-*+]\s+/gm, "")
      .replace(/^\s*\d+\.\s+/gm, "");
  };

  // Component return: conditionally render if not clusters
  if (activeTab !== "clusters") {
    return (
      <div className="min-h-screen bg-gray-50 p-6 pt-24">
        {/* Logo at top left with good padding - persistent across all tabs */}
        <div className="absolute top-8 left-10 z-10">
          <h1 className="font-['var(--font-playfair-display)'] text-6xl font-bold text-black drop-shadow-md">
            Overseer
          </h1>
        </div>
        {/* Empty div instead of placeholder text */}
      </div>
    );
  }

  // Otherwise, show the 3D canvas
  return (
    <div className="relative w-full h-full" ref={containerRef}>
      <canvas ref={canvasRef} className="w-full h-full" />

      {/* Hover Information Box - positioned based on sphere location */}
      {hoveredNode && hoverPosition && (
        <div
          className="absolute z-10 bg-white/90 backdrop-blur-sm p-3 rounded-lg shadow-md max-w-xs"
          style={{
            left: `${hoverPosition.x}px`,
            top: `${hoverPosition.y}px`,
            transform: "translate(0, -100%)",
          }}
        >
          <h3 className="text-sm font-semibold text-gray-800">
            Node Information
          </h3>
          <div className="text-sm text-gray-600 mt-1">
            <p>ID: {hoveredNode.id}</p>
            <p>Cluster: {hoveredNode.cluster_id}</p>
          </div>
        </div>
      )}

      {/* Increase top padding further to position under the title */}
      <div className="absolute top-28 left-10 z-10 space-y-4">
        {/* Title removed from here - now in page.tsx */}

        {/* Cluster Panel */}
        {clusterEmbeddings && activeTab === "clusters" && (
          <div className="bg-white/80 backdrop-blur-sm p-4 rounded-lg shadow-md max-w-xs">
            <h3 className="text-sm font-semibold text-gray-800 mb-3">
              Clusters
            </h3>

            {/* Rest of cluster panel content */}
            <div className="space-y-2.5 max-h-[60vh] overflow-y-auto pr-1">
              {/* Cluster items */}
              {Object.entries(clusterEmbeddings?.clusters || {}).map(
                ([clusterId, clusterInfo]: [string, any], index) => {
                  const clusterColor =
                    clusterColors[index % clusterColors.length];
                  // Use letters instead of cluster IDs
                  const clusterLetter = String.fromCharCode(65 + (index % 26));
                  return (
                    <div
                      key={clusterId}
                      className="flex items-center space-x-2 text-black py-0.5"
                    >
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id={`cluster-${clusterId}`}
                          checked={visibleClusters[clusterId] !== false}
                          onChange={() => toggleClusterVisibility(clusterId)}
                          className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                      </div>
                      <label
                        htmlFor={`cluster-${clusterId}`}
                        className="flex-grow flex items-center cursor-pointer text-xs text-black"
                      >
                        <span
                          className="inline-block w-3 h-3 mr-1.5 rounded-full"
                          style={{
                            backgroundColor: `#${clusterColor.getHexString()}`,
                          }}
                        ></span>
                        Cluster {clusterLetter}
                        {clusterInfo.size && (
                          <span className="text-xs text-gray-500 ml-1">
                            ({clusterInfo.size})
                          </span>
                        )}
                      </label>
                      <button
                        onClick={() => selectCluster(clusterId)}
                        className="ml-auto text-xs text-blue-500 hover:text-blue-700 font-medium"
                      >
                        {selectedCluster === clusterId ? "Hide" : "Info"}
                      </button>
                    </div>
                  );
                }
              )}
            </div>

            {/* Recenter button */}
            <button
              onClick={handleRecenter}
              className="mt-2.5 w-full py-2 bg-blue-50 hover:bg-blue-100 text-blue-600 font-medium rounded text-xs transition-colors duration-200 flex justify-center items-center"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4 mr-1.5"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M4 2a2 2 0 00-2 2v11a3 3 0 106 0V4a2 2 0 00-2-2H4zm1 14a1 1 0 100-2 1 1 0 000 2zm5-1.757l4.9-4.9a2 2 0 000-2.828L13.485 5.1a2 2 0 00-2.828 0L10 5.757v8.486zM16 18H9.071l6-6H16a2 2 0 012 2v2a2 2 0 01-2 2z"
                  clipRule="evenodd"
                />
              </svg>
              Recenter View
            </button>
          </div>
        )}

        {/* Cluster Analysis Panel - smaller */}
        {selectedCluster && (
          <div className="bg-white/80 backdrop-blur-sm p-3 rounded-lg shadow-md max-h-[50vh] max-w-xs overflow-y-auto">
            <div className="flex justify-between items-center mb-3 pb-2 border-b border-gray-200 bg-blue-50 -m-3 p-3 rounded-t-lg">
              <h3 className="text-base font-bold text-blue-700">
                Cluster {getClusterLetter(selectedCluster, clusterEmbeddings)}{" "}
                Analysis
              </h3>
              <button
                onClick={() => setSelectedCluster(null)}
                className="text-gray-500 hover:text-gray-700 text-sm"
              >
                <span className="sr-only">Close</span>✕
              </button>
            </div>
            <div className="text-xs text-gray-600 whitespace-pre-wrap leading-relaxed mt-2">
              {clusterAnalyses[selectedCluster]
                ? cleanAnalysisText(clusterAnalyses[selectedCluster])
                : "Loading analysis..."}
            </div>
          </div>
        )}
      </div>

      {/* Information box in bottom right corner */}
      <div className="absolute bottom-4 right-4 bg-gray-700/80 text-white p-3 rounded-lg shadow-lg max-w-xs">
        <p className="text-sm">Click on spheres to view details</p>
        <p className="text-sm">Drag to rotate | Scroll to zoom</p>
      </div>
    </div>
  );
}
