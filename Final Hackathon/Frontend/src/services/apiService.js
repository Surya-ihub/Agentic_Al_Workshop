// src/api.js
import axios from "axios";

export const baseURL = "http://localhost:5000"

const api = axios.create({
  baseURL: baseURL // Adjust if hosted elsewhere
});

export default api;

// Fetch latest skill analysis for a user
export const fetchSkillAnalysis = async (userId) => {
  const res = await api.get(`/api/skillgap/data/${userId}`);
  return res.data.data;
};
