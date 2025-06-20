import axios from "axios";
import React, { useState } from "react";
import { fetchSkillAnalysis } from "../../services/apiService";
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip, Legend,
} from "recharts";

export default function SkillSignalUploader() {
  const [jobRole, setJobRole] = useState("");
  const [jsonData, setJsonData] = useState(null);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [analysisData, setAnalysisData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  // Track completed modules (no localStorage)
  const [completedModules, setCompletedModules] = useState([]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target.result);
        setJsonData(parsed);
        setError("");
        setSuccess("");
      } catch (err) {
        setError("Invalid JSON file.");
        setJsonData(null);
        setSuccess("");
      }
    };
    reader.readAsText(file);
  };

  const handleRoleChange = (e) => {
    setJobRole(e.target.value);
    setSuccess("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!jsonData) {
      setError("Please upload a valid JSON file.");
      return;
    }
    if (!jobRole) {
      setError("Please select a job role.");
      return;
    }

    const userId = localStorage.getItem("userId")
    if (!userId) {
      setError("JSON file must include 'userId' field.");
      return;
    }

    setIsLoading(true);
    try {
      const res = await axios.post(`http://localhost:5000/api/skillgap/data/${userId}`, {
        jobRole,
        data: jsonData,
      });

      setError("");
      setSuccess("Data submitted successfully!");

      // Use the new response structure directly
      setAnalysisData(res.data);
    } catch (err) {
      console.error(err);
      setError("Failed to submit data.");
      setSuccess("");
    } finally {
      setIsLoading(false);
    }
  };

  // Helper: Format radar chart data
  const getRadarData = () => {
    if (!analysisData?.skillProgressTracker?.radar_chart_data) return [];
    return analysisData.skillProgressTracker.radar_chart_data.map((item) => ({
      skill: item.skill,
      level: item.level_num,
    }));
  };

  // Helper: Deficiency dashboard rows
  const getDeficiencyRows = () => {
    const gapReport = analysisData?.skillProgressTracker?.gap_report || analysisData?.skillGapAnalysis?.gap_report || [];
    return gapReport.length ? gapReport : (analysisData?.skillGapAnalysis?.missing_core_skills || []).map(skill => ({ skill, type: "core" }));
  };

  // Helper: Remediation plan rows
  const getRemediationRows = () => {
    const roadmap = analysisData?.remediationPlan?.remediation_roadmap || analysisData?.remediation_roadmap || [];
    return roadmap;
  };

  const handleMarkComplete = async (mod) => {
    if (completedModules.includes(mod.module_id)) return;
    const updated = [...completedModules, mod.module_id];
    setCompletedModules(updated);
    // Optionally, send to backend
    try {
      const userId = localStorage.getItem("userId");
      await axios.post("http://localhost:5000/api/skillgap/mark-module-complete", {
        userId,
        moduleId: mod.module_id,
        skill: mod.skill,
      });
    } catch (err) {
      // Optionally handle error
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-gray-100">
      <div className="bg-white rounded-2xl shadow-lg p-6 w-full max-w-4xl">
        {/* Introductory Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2 text-center text-blue-700">
            Agentic Skill Deficiency Auditor
          </h1>
          <p className="text-center text-gray-700 mb-4 max-w-2xl mx-auto">
            <span className="font-semibold text-blue-800">How it works:</span> Upload your <span className="font-semibold">learner analytics JSON</span> (containing code submissions, quiz results, project feedback, etc.) and select your target job role. Our AI-powered multi-agent system will:
          </p>
          <ul className="list-disc text-gray-700 text-sm max-w-2xl mx-auto mb-2 pl-6">
            <li>Analyze your skills and build a dynamic skill graph</li>
            <li>Benchmark your profile against live job market requirements</li>
            <li>Detect and prioritize your most critical skill gaps</li>
            <li>Generate a personalized remediation roadmap (with micro-learning modules)</li>
            <li>Visualize your readiness with an interactive radar chart and progress dashboard</li>
          </ul>
          <p className="text-center text-gray-600 text-xs max-w-2xl mx-auto">
            <span className="font-semibold">Output:</span> You'll see a skill radar chart, a deficiency dashboard, and a remediation progress view—giving you a clear, actionable path to career alignment.
          </p>
        </div>

        {/* Upload Interface */}
        <div className="mb-8 border-b pb-6">
          <h2 className="text-md font-[500] mb-4 text-blue-900">Upload Learner Data & Select Job Role</h2>
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div>
              <label className="block mb-1 font-medium">Upload JSON File</label>
              <input
                type="file"
                accept=".json"
                onChange={handleFileUpload}
                className="border border-gray-300 p-2 rounded w-full"
                disabled={isLoading}
              />
            </div>
            <div>
              <label className="block mb-1 font-medium">Select Job Role</label>
              <select
                value={jobRole}
                onChange={handleRoleChange}
                className="border border-gray-300 p-2 rounded w-full"
                disabled={isLoading}
              >
                <option value="">-- Select Role --</option>
                <option value="frontend">Frontend</option>
                <option value="backend">Backend</option>
              </select>
            </div>
            {error && <p className="text-red-500 mb-2">{error}</p>}
            {success && <p className="text-green-600 mb-2">{success}</p>}
            <button 
              type="submit"
              disabled={isLoading}
              className={`px-4 py-2 rounded w-full font-semibold flex items-center justify-center transition-colors ${
                isLoading 
                  ? 'bg-blue-400 cursor-not-allowed' 
                  : 'bg-blue-600 hover:bg-blue-700'
              } text-white`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </>
              ) : (
                'Submit'
              )}
            </button>
          </form>
        </div>

        {/* Skill Radar Chart */}
        {analysisData?.skillProgressTracker?.radar_chart_data && (
          <div className="mt-8 border-b pb-8">
            <h2 className="text-xl font-bold mb-4 text-blue-800 text-center">Skill Radar Chart</h2>
            <div className="flex flex-col items-center">
              <ResponsiveContainer width="100%" minWidth={350} height={400}>
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={getRadarData()}>
                  <PolarGrid stroke="#e5e7eb" />
                  <PolarAngleAxis dataKey="skill" tick={{ fontSize: 13, fill: '#1e293b' }} />
                  <PolarRadiusAxis angle={30} domain={[0, 4]} tickCount={5} tickFormatter={v => ['Novice','Beginner','Intermediate','Proficient','Advanced'][v] || v} />
                  <Radar name="Skill Level" dataKey="level" stroke="#2563eb" fill="#60a5fa" fillOpacity={0.7} />
                  <Tooltip formatter={(value) => [value, "Level"]} />
                  <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ color: '#2563eb' }} />
                </RadarChart>
              </ResponsiveContainer>
              <div className="mt-2 text-xs text-gray-500">Level: 0=Novice, 1=Beginner, 2=Intermediate, 3=Proficient, 4=Advanced</div>
            </div>
          </div>
        )}

        {/* Deficiency Dashboard */}
        {getDeficiencyRows().length > 0 && (
          <div className="mt-8 border-b pb-8">
            <h2 className="text-xl font-bold mb-4 text-red-800 text-center">Deficiency Dashboard</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white border border-gray-200 rounded">
                <thead>
                  <tr>
                    <th className="px-3 py-2 border-b">Skill</th>
                    <th className="px-3 py-2 border-b">Type</th>
                    <th className="px-3 py-2 border-b">Urgency</th>
                    <th className="px-3 py-2 border-b">Impact</th>
                    <th className="px-3 py-2 border-b">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {getDeficiencyRows().map((row, idx) => (
                    <tr key={idx} className="text-sm">
                      <td className="px-3 py-2 border-b">{row.skill}</td>
                      <td className="px-3 py-2 border-b">{row.type}</td>
                      <td className="px-3 py-2 border-b">{row.urgency || '-'}</td>
                      <td className="px-3 py-2 border-b">{row.career_impact || '-'}</td>
                      <td className="px-3 py-2 border-b">{row.reason || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Remediation Progress View */}
        {getRemediationRows().length > 0 && (
          <div className="mt-8">
            <div className="flex items-center justify-center mb-6">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <h2 className="text-2xl font-bold text-green-800">Remediation Progress</h2>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              </div>
            </div>
            
            <div className="space-y-4">
              {getRemediationRows().map((day, idx) => (
                <div key={idx} className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow duration-200">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="flex items-center justify-center w-8 h-8 bg-green-600 text-white rounded-full font-bold text-sm">
                      {idx + 1}
                    </div>
                    <h3 className="text-lg font-bold text-green-800">{day.day}</h3>
                    <div className="flex-1 h-px bg-gradient-to-r from-green-300 to-transparent"></div>
                    <span className="text-sm text-gray-600 bg-white px-2 py-1 rounded-full border">
                      {day.modules.length} module{day.modules.length !== 1 ? 's' : ''}
                    </span>
                  </div>
                  
                  <div className="grid gap-3 ml-11">
                    {day.modules.map((mod, i) => (
                      <div key={i} className={`bg-white rounded-lg p-3 border transition-colors duration-150 ${
                        completedModules.includes(mod.module_id)
                          ? 'border-green-400 bg-green-50 opacity-60'
                          : 'border-gray-200 hover:border-blue-300'
                      }`}>
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-semibold text-blue-900 mb-1">{mod.title}</h4>
                            <div className="flex items-center gap-4 text-sm text-gray-600">
                              <span className="flex items-center gap-1">
                                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                                <span className="font-medium">{mod.skill}</span>
                              </span>
                              <span className="flex items-center gap-1">
                                <svg className="w-3 h-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span>{mod.duration_min} min</span>
                              </span>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 ml-4">
                            <button
                              className={`w-4 h-4 border-2 rounded transition-colors duration-150 flex items-center justify-center ${
                                completedModules.includes(mod.module_id)
                                  ? 'border-green-500 bg-green-400'
                                  : 'border-gray-300 hover:border-green-500'
                              }`}
                              onClick={() => handleMarkComplete(mod)}
                              disabled={completedModules.includes(mod.module_id)}
                              title={completedModules.includes(mod.module_id) ? 'Completed' : 'Mark Complete'}
                            >
                              {completedModules.includes(mod.module_id) && (
                                <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                </svg>
                              )}
                            </button>
                            <span className={`text-xs ${completedModules.includes(mod.module_id) ? 'text-green-600' : 'text-gray-400'}`}>
                              {completedModules.includes(mod.module_id) ? 'Completed' : 'Mark Complete'}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {/* Progress indicator */}
                  <div className="mt-4 ml-11">
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span>Progress:</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div className="bg-gradient-to-r from-green-400 to-blue-500 h-2 rounded-full" style={{width: '0%'}}></div>
                      </div>
                      <span>0/{day.modules.length}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            {/* Summary Stats */}
            <div className="mt-6 bg-gradient-to-r from-green-100 to-blue-100 rounded-lg p-4 border border-green-200">
              <h4 className="font-semibold text-green-800 mb-2">Remediation Summary</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-gray-700">
                    Total Days: <span className="font-semibold text-green-700">{getRemediationRows().length}</span>
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span className="text-gray-700">
                    Total Modules: <span className="font-semibold text-blue-700">
                      {getRemediationRows().reduce((acc, day) => acc + day.modules.length, 0)}
                    </span>
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  <span className="text-gray-700">
                    Estimated Time: <span className="font-semibold text-purple-700">
                      {getRemediationRows().reduce((acc, day) => 
                        acc + day.modules.reduce((dayAcc, mod) => dayAcc + mod.duration_min, 0), 0
                      )} min
                    </span>
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}