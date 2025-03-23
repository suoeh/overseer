// API client for interacting with the backend
const API_BASE_URL = "http://localhost:3002/api";

// Ensure we don't have trailing slashes that might cause double-slash issues
const getApiUrl = (endpoint) => {
  // Remove any leading slash from the endpoint to avoid double slashes
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint.substring(1) : endpoint;
  return `${API_BASE_URL}/${cleanEndpoint}`;
};

/**
 * Check if the API is running
 * @returns {Promise<Object>} Status information
 */
export const getApiStatus = async () => {
  try {
    const response = await fetch(getApiUrl('health'));
    return await response.json();
  } catch (error) {
    console.error("Error checking API status:", error);
    return { status: "error", message: error.message };
  }
};

/**
 * Upload a CSV file to use as the dataset for unbiasing
 * @param {File} file - The CSV file to upload
 * @returns {Promise<Object>} Upload result with job ID
 */
export const uploadDataset = async (file, clusterCount) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('cluster_count', clusterCount);
    
    const response = await fetch(getApiUrl('upload'), {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Upload failed');
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error uploading dataset:", error);
    throw error;
  }
};

/**
 * Check the status of a processing job
 * @param {string} jobId - The job ID to check
 * @returns {Promise<Object>} Job status information
 */
export const getJobStatus = async (jobId) => {
  try {
    const response = await fetch(getApiUrl(`jobs/${jobId}/status`));
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to get job status');
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Error getting job status for ${jobId}:`, error);
    throw error;
  }
};

/**
 * Get information about which datasets are available
 * @returns {Promise<Object>} Available datasets information
 */
export const getAvailableDatasets = async () => {
  try {
    const response = await fetch(getApiUrl('datasets/available'));
    return await response.json();
  } catch (error) {
    console.error("Error fetching available datasets:", error);
    return {};
  }
};

/**
 * Fetch a page of data from any paginated dataset endpoint
 * @param {string} endpoint - API endpoint path without the base URL
 * @param {number} page - Page number (default: 1)
 * @param {number} pageSize - Number of records per page (default: 100)
 * @returns {Promise<Object>} Paginated data
 */
export const fetchPaginatedDataset = async (endpoint, page = 1, pageSize = 100) => {
  try {
    const url = new URL(getApiUrl(endpoint));
    url.searchParams.append("page", page);
    url.searchParams.append("page_size", pageSize);
    
    const response = await fetch(url);
    
    if (response.status === 404) {
      console.warn(`Dataset not found: ${endpoint}`);
      return {};
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching ${endpoint}:`, error);
    return {};
  }
};

/**
 * Get a page of the cleaned resumes dataset
 * @param {number} page - Page number
 * @param {number} pageSize - Number of records per page
 * @returns {Promise<Object>} Paginated data
 */
export const getCleanedResumes = (page = 1, pageSize = 100) => {
  return fetchPaginatedDataset("cleaned_resumes", page, pageSize);
};

/**
 * Get a page of the unbiased resumes dataset
 * @param {number} page - Page number
 * @param {number} pageSize - Number of records per page
 * @returns {Promise<Object>} Paginated data
 */
export const getUnbiasedResumes = (page = 1, pageSize = 100) => {
  return fetchPaginatedDataset("unbiased_resumes", page, pageSize);
};

/**
 * Get a page of the removed entries dataset
 * @param {number} page - Page number
 * @param {number} pageSize - Number of records per page
 * @returns {Promise<Object>} Paginated data
 */
export const getRemovedEntries = (page = 1, pageSize = 100) => {
  return fetchPaginatedDataset("removed_entries", page, pageSize);
};

/**
 * Get a page of the all_clusters dataset
 * @param {number} page - Page number
 * @param {number} pageSize - Number of records per page
 * @returns {Promise<Object>} Paginated data
 */
export const getAllClustersDataset = (page = 1, pageSize = 100) => {
  return fetchPaginatedDataset("all_clusters", page, pageSize);
};

/**
 * Get information about all clusters
 * @returns {Promise<Object>} All clusters information
 */
export const getAllClustersInfo = async () => {
  try {
    const response = await fetch(getApiUrl("clusters"));
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to fetch cluster embeddings: ${errorText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error fetching cluster embeddings:", error);
    throw error;
  }
};

/**
 * Get a specific cluster by ID
 * @param {number} clusterId - Cluster ID
 * @param {number} page - Page number
 * @param {number} pageSize - Number of records per page
 * @returns {Promise<Object>} Cluster data
 */
export const getCluster = async (clusterId, page = 1, pageSize = 100) => {
  try {
    const url = new URL(getApiUrl(`clusters/${clusterId}`));
    url.searchParams.append("page", page);
    url.searchParams.append("page_size", pageSize);
    
    const response = await fetch(url);
    
    if (response.status === 404) {
      console.warn(`Cluster ${clusterId} not found`);
      return {};
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Error fetching cluster ${clusterId}:`, error);
    return {};
  }
};

/**
 * Get all cluster analyses
 * @returns {Promise<Object>} All analyses
 */
export const getAllClusterAnalyses = async () => {
  try {
    const response = await fetch(getApiUrl('analysis/clusters'));
    
    if (response.status === 404) {
      console.warn("Cluster analyses not found");
      return {};
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error fetching cluster analyses:", error);
    return {};
  }
};

/**
 * Fetches the unbiased embeddings data for console logging
 * @returns {Promise<Object>} The embeddings data
 */
export const getUnbiasedEmbeddingsData = async () => {
    try {
      const response = await fetch(getApiUrl("unbiased_embeddings_data"));
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch unbiased embeddings: ${errorText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error("Error fetching unbiased embeddings data:", error);
      throw error;
    }
  };

/**
 * Get analysis for a specific cluster
 * @param {number} clusterId - Cluster ID
 * @returns {Promise<Object>} Cluster analysis
 */
export const getClusterAnalysis = async (clusterId) => {
  try {
    const response = await fetch(getApiUrl(`analysis/clusters/${clusterId}`));
    
    if (response.status === 404) {
      console.warn(`Analysis for cluster ${clusterId} not found`);
      return {};
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Error fetching analysis for cluster ${clusterId}:`, error);
    return {};
  }
};

/**
 * Get the unbiasing summary
 * @returns {Promise<Object>} Unbiasing summary
 */
export const getUnbiasingSummary = async () => {
  try {
    const response = await fetch(getApiUrl('summary'));
    
    if (response.status === 404) {
      console.warn("Unbiasing summary not found");
      return {};
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error fetching unbiasing summary:", error);
    return {};
  }
};

/**
 * Download a file from the API
 * @param {string} fileType - Type of file to download (e.g., "cleaned_resumes", "unbiased_resumes")
 * @returns {Promise<Blob>} File blob that can be saved using FileSaver or similar
 */
export const downloadFile = async (fileType) => {
  try {
    const response = await fetch(getApiUrl(`download/${fileType}`));
    
    if (response.status === 404) {
      console.warn(`File ${fileType} not found`);
      return null;
    }
    
    return await response.blob();
  } catch (error) {
    console.error(`Error downloading ${fileType}:`, error);
    return null;
  }
};

/**
 * Save a blob as a file in the browser
 * @param {Blob} blob - File blob
 * @param {string} filename - Filename to save as
 */
export const saveFile = (blob, filename) => {
  if (!blob) return;
  
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
};

/**
 * Fetch all dataset statistics
 * @returns {Promise<Object>} Statistics for all available datasets
 */
export const fetchAllDatasetStats = async () => {
  try {
    const available = await getAvailableDatasets();
    const stats = {};
    
    if (available.cleaned_resumes) {
      const data = await getCleanedResumes(1, 1);
      stats.cleanedResumes = {
        totalRecords: data.total_records || 0,
        totalPages: data.total_pages || 0
      };
    }
    
    if (available.unbiased_resumes) {
      const data = await getUnbiasedResumes(1, 1);
      stats.unbiasedResumes = {
        totalRecords: data.total_records || 0,
        totalPages: data.total_pages || 0
      };
    }
    
    if (available.removed_entries) {
      const data = await getRemovedEntries(1, 1);
      stats.removedEntries = {
        totalRecords: data.total_records || 0,
        totalPages: data.total_pages || 0
      };
    }
    
    if (available.all_clusters) {
      const data = await getAllClustersDataset(1, 1);
      stats.allClusters = {
        totalRecords: data.total_records || 0,
        totalPages: data.total_pages || 0
      };
    }
    
    if (available.individual_clusters && available.individual_clusters.length > 0) {
      const clustersInfo = await getAllClustersInfo();
      stats.clusters = {
        totalClusters: clustersInfo.total_clusters || 0,
        clusterSizes: Object.entries(clustersInfo.clusters || {}).reduce((acc, [key, value]) => {
          acc[key] = value.total_records || 0;
          return acc;
        }, {})
      };
    }
    
    return stats;
  } catch (error) {
    console.error("Error fetching dataset stats:", error);
    return {};
  }
};

/**
 * Fetches the removed embeddings data for console logging
 * @returns {Promise<Object>} The embeddings data
 */
export const getRemovedEmbeddingsData = async () => {
    try {
      const response = await fetch(getApiUrl("removed_embeddings_data"));
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch removed embeddings: ${errorText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error("Error fetching removed embeddings data:", error);
      throw error;
    }
  }; 