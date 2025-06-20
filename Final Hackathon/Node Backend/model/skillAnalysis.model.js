import mongoose from "mongoose";

const skillAnalysisSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
  },
  skillAssessment: {
    type: mongoose.Schema.Types.Mixed,
    default: null,
  },
  jobSearchResults: {
    type: mongoose.Schema.Types.Mixed,
    default: null,
  },
  roleAnalysis: {
    type: mongoose.Schema.Types.Mixed,
    default: null,
  },
  skillGapAnalysis: {
    type: mongoose.Schema.Types.Mixed,
    default: null,
  },
  remediationPlan: {
    type: mongoose.Schema.Types.Mixed,
    default: null,
  },
  skillProgressTracker: {
    type: mongoose.Schema.Types.Mixed,
    default: null,
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: null,
  },
  response: {
    type: mongoose.Schema.Types.Mixed, // Store the full FastAPI response for backward compatibility
    required: true,
  },
}, { timestamps: true });

export const SkillAnalysis = mongoose.model("SkillAnalysis", skillAnalysisSchema); 