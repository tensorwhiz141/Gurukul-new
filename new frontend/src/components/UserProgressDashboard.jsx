import React from "react";
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  BookOpen, 
  Target,
  Brain,
  Zap
} from "lucide-react";
import GlassContainer from "./GlassContainer";
import GlassButton from "./GlassButton";

const UserProgressDashboard = ({ 
  userProgress, 
  userAnalytics, 
  onTriggerIntervention, 
  isLoadingProgress = false,
  isLoadingAnalytics = false,
  isTriggeringIntervention = false 
}) => {
  if (isLoadingProgress || isLoadingAnalytics) {
    return (
      <GlassContainer className="p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-white/20 rounded mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-white/10 rounded"></div>
            <div className="h-4 bg-white/10 rounded w-3/4"></div>
            <div className="h-4 bg-white/10 rounded w-1/2"></div>
          </div>
        </div>
      </GlassContainer>
    );
  }

  if (!userProgress && !userAnalytics) {
    return (
      <GlassContainer className="p-6 text-center">
        <Brain className="w-12 h-12 text-white/50 mx-auto mb-3" />
        <p className="text-white/70">No progress data available yet.</p>
        <p className="text-white/50 text-sm mt-2">
          Complete some lessons to see your learning analytics!
        </p>
      </GlassContainer>
    );
  }

  const progressData = userProgress || {};
  const analyticsData = userAnalytics || {};
  
  // Extract key metrics
  const quizScores = progressData.quiz_scores || analyticsData.orchestration_data?.orchestration_session?.educational_progress?.quiz_scores || [];
  const averageScore = quizScores.length > 0 ? quizScores.reduce((a, b) => a + b, 0) / quizScores.length : null;
  const latestScore = quizScores.length > 0 ? quizScores[quizScores.length - 1] : null;
  const lessonCount = analyticsData.lesson_count || 0;
  const triggerCount = analyticsData.trigger_count || 0;
  const recommendations = progressData.recommendations || [];
  const triggers = progressData.triggers_detected || [];

  // Performance status
  const getPerformanceStatus = () => {
    if (!averageScore) return { status: 'unknown', color: 'text-gray-400', icon: BookOpen };
    if (averageScore >= 80) return { status: 'excellent', color: 'text-green-400', icon: CheckCircle };
    if (averageScore >= 70) return { status: 'good', color: 'text-blue-400', icon: TrendingUp };
    if (averageScore >= 60) return { status: 'average', color: 'text-yellow-400', icon: Target };
    return { status: 'needs_help', color: 'text-red-400', icon: AlertTriangle };
  };

  const performanceStatus = getPerformanceStatus();
  const StatusIcon = performanceStatus.icon;

  // Trend calculation
  const getTrend = () => {
    if (quizScores.length < 2) return null;
    const recent = quizScores.slice(-3);
    const older = quizScores.slice(-6, -3);
    if (older.length === 0) return null;
    
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
    
    return recentAvg > olderAvg ? 'improving' : recentAvg < olderAvg ? 'declining' : 'stable';
  };

  const trend = getTrend();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold text-white flex items-center">
          <Brain className="w-8 h-8 mr-3 text-blue-400" />
          Learning Progress
        </h3>
        {triggers.length > 0 && (
          <GlassButton
            onClick={onTriggerIntervention}
            disabled={isTriggeringIntervention}
            className="bg-amber-500/20 hover:bg-amber-500/30 border-amber-500/40 text-amber-200"
          >
            {isTriggeringIntervention ? (
              <>
                <div className="animate-spin w-4 h-4 border-2 border-amber-300 border-t-transparent rounded-full mr-2"></div>
                Triggering...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Get Help ({triggers.length})
              </>
            )}
          </GlassButton>
        )}
      </div>

      {/* Performance Overview */}
      <GlassContainer className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Overall Performance */}
          <div className="text-center">
            <StatusIcon className={`w-12 h-12 mx-auto mb-3 ${performanceStatus.color}`} />
            <h4 className="text-lg font-semibold text-white mb-2">Overall Performance</h4>
            <p className={`text-2xl font-bold ${performanceStatus.color}`}>
              {averageScore ? `${averageScore.toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-white/60 text-sm capitalize">{performanceStatus.status.replace('_', ' ')}</p>
          </div>

          {/* Learning Activity */}
          <div className="text-center">
            <BookOpen className="w-12 h-12 mx-auto mb-3 text-blue-400" />
            <h4 className="text-lg font-semibold text-white mb-2">Lessons Completed</h4>
            <p className="text-2xl font-bold text-blue-400">{lessonCount}</p>
            <p className="text-white/60 text-sm">Total lessons</p>
          </div>

          {/* Trend */}
          <div className="text-center">
            {trend === 'improving' ? (
              <TrendingUp className="w-12 h-12 mx-auto mb-3 text-green-400" />
            ) : trend === 'declining' ? (
              <TrendingDown className="w-12 h-12 mx-auto mb-3 text-red-400" />
            ) : (
              <Target className="w-12 h-12 mx-auto mb-3 text-gray-400" />
            )}
            <h4 className="text-lg font-semibold text-white mb-2">Trend</h4>
            <p className={`text-2xl font-bold ${
              trend === 'improving' ? 'text-green-400' : 
              trend === 'declining' ? 'text-red-400' : 'text-gray-400'
            }`}>
              {trend ? trend.charAt(0).toUpperCase() + trend.slice(1) : 'Stable'}
            </p>
            <p className="text-white/60 text-sm">Recent performance</p>
          </div>
        </div>
      </GlassContainer>

      {/* Quiz Scores History */}
      {quizScores.length > 0 && (
        <GlassContainer className="p-6">
          <h4 className="text-lg font-semibold text-white mb-4">Recent Quiz Scores</h4>
          <div className="flex items-end space-x-2 h-32">
            {quizScores.slice(-10).map((score, index) => (
              <div key={index} className="flex-1 flex flex-col items-center">
                <div 
                  className={`w-full rounded-t transition-all duration-300 ${
                    score >= 80 ? 'bg-green-500' :
                    score >= 70 ? 'bg-blue-500' :
                    score >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ height: `${(score / 100) * 100}%` }}
                ></div>
                <span className="text-xs text-white/60 mt-1">{score}%</span>
              </div>
            ))}
          </div>
        </GlassContainer>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <GlassContainer className="p-6">
          <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-blue-400" />
            Personalized Recommendations
          </h4>
          <div className="space-y-3">
            {recommendations.slice(0, 3).map((rec, index) => (
              <div 
                key={index}
                className={`p-4 rounded-lg border-l-4 ${
                  rec.priority === 'high' ? 'bg-red-900/20 border-red-500' :
                  rec.priority === 'medium' ? 'bg-yellow-900/20 border-yellow-500' :
                  'bg-blue-900/20 border-blue-500'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-white font-medium">{rec.message}</p>
                    <p className="text-white/60 text-sm mt-1 capitalize">
                      {rec.type.replace('_', ' ')} • {rec.priority} priority
                    </p>
                  </div>
                  {rec.priority === 'high' && (
                    <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 ml-3" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </GlassContainer>
      )}

      {/* Triggers Alert */}
      {triggers.length > 0 && (
        <GlassContainer className="p-6 bg-amber-900/20 border-amber-500/40">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <AlertTriangle className="w-6 h-6 text-amber-400 mr-3" />
              <div>
                <h4 className="text-lg font-semibold text-amber-200">Intervention Recommended</h4>
                <p className="text-amber-300/80">
                  {triggers.length} trigger{triggers.length > 1 ? 's' : ''} detected that may need attention
                </p>
              </div>
            </div>
            <GlassButton
              onClick={onTriggerIntervention}
              disabled={isTriggeringIntervention}
              className="bg-amber-500/20 hover:bg-amber-500/30 border-amber-500/40 text-amber-200"
            >
              {isTriggeringIntervention ? 'Processing...' : 'Get Support'}
            </GlassButton>
          </div>
        </GlassContainer>
      )}
    </div>
  );
};

export default UserProgressDashboard;
