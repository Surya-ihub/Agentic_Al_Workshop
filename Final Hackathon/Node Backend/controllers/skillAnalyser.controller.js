import axios from "axios";
import mongoose from "mongoose";
import { SkillAnalysis } from "../model/skillAnalysis.model.js";

// Helper to clean Python code block or variable assignment and return JSON
function cleanPythonDictString(raw) {
    if (typeof raw !== 'string') return raw;
    // Remove code block markers and variable assignment
    let cleaned = raw.replace(/```[a-zA-Z]*\n?/g, '')
                     .replace(/^\s*\w+\s*=\s*/, '')
                     .replace(/```/g, '')
                     .trim();
    // Try to parse as JSON
    try {
        return JSON.parse(cleaned);
    } catch {
        // Try to parse as JS object if JSON fails
        try {
            // eslint-disable-next-line no-eval
            return eval('(' + cleaned + ')');
        } catch {
            return raw; // fallback to raw if all fails
        }
    }
}

// Recursively clean all string fields that look like Python dicts/code blocks
function deepCleanResponse(obj) {
    if (Array.isArray(obj)) {
        return obj.map(deepCleanResponse);
    } else if (obj && typeof obj === 'object') {
        const cleaned = {};
        for (const key in obj) {
            if (typeof obj[key] === 'string' && obj[key].includes('{') && obj[key].includes('}')) {
                cleaned[key] = cleanPythonDictString(obj[key]);
            } else {
                cleaned[key] = deepCleanResponse(obj[key]);
            }
        }
        return cleaned;
    } else {
        return obj;
    }
}

export const postFeedback = async (req, res) => {
    try {
        const { jobRole, data } = req.body;
        const { userId } = req.params

        // Forward to FastAPI
        const fastapiURL = `http://localhost:8000/api/skillgap/data/${userId}`;
        const response = await axios.post(fastapiURL, {
            role: jobRole,
            data: data,
        });
        console.log(response.data, 15)

        // Clean the FastAPI response recursively
        const cleanedResponse = deepCleanResponse(response.data);

        // Store the cleaned FastAPI response in MongoDB
        const userObjectId = mongoose.Types.ObjectId.isValid(userId) ? new mongoose.Types.ObjectId(userId) : userId;
        // Extract main fields for structured storage
        const {
          skillAssessment,
          jobSearchResults,
          roleAnalysis,
          skillGapAnalysis,
          remediationPlan,
          skillProgressTracker,
          metadata
        } = cleanedResponse;
        await SkillAnalysis.findOneAndUpdate(
            { userId: userObjectId },
            {
                skillAssessment,
                jobSearchResults,
                roleAnalysis,
                skillGapAnalysis,
                remediationPlan,
                skillProgressTracker,
                metadata,
                response: cleanedResponse
            },
            { upsert: true, new: true, setDefaultsOnInsert: true }
        );

        return res.status(200).json({
            message: "Forwarded to FastAPI",
            skillAssessment,
            jobSearchResults,
            roleAnalysis,
            skillGapAnalysis,
            remediationPlan,
            skillProgressTracker,
            metadata
        });
    } catch (error) {
        console.error("❌ FastAPI forwarding error:", error.message);
        return res.status(500).json({ message: "Server error forwarding to FastAPI" });
    }
};

export const getSkillAnalysis = async (req, res) => {
    try {
        const { userId } = req.params;
        const userObjectId = mongoose.Types.ObjectId.isValid(userId) ? new mongoose.Types.ObjectId(userId) : userId;
        const record = await SkillAnalysis.findOne({ userId: userObjectId }).sort({ updatedAt: -1 });
        if (!record) {
            return res.status(404).json({ message: "No skill analysis found for this user." });
        }
        return res.status(200).json({
            message: "Skill analysis fetched successfully",
            data: record.response,
        });
    } catch (error) {
        console.error("❌ Error fetching skill analysis:", error.message);
        return res.status(500).json({ message: "Server error fetching skill analysis" });
    }
};
