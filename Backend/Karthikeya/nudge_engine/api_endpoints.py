"""
API Endpoints Module for Karthikeya Multilingual Reporting Engine
Flask-based REST API endpoints for report and nudge generation
"""

import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from .sentiment_analyzer import SentimentAnalyzer
from .report_generator import ReportGenerator, ReportData
from .config_loader import ConfigLoader
from .template_loader import TemplateLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(config_dir: str = "config", templates_dir: str = "templates") -> Flask:
    """
    Create and configure Flask application
    
    Args:
        config_dir: Directory containing configuration files
        templates_dir: Directory containing template files
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    CORS(app)
    
    # Initialize components
    config_loader = ConfigLoader(config_dir)
    template_loader = TemplateLoader(templates_dir)
    sentiment_analyzer = SentimentAnalyzer(
        config_path=f"{templates_dir}/sentiment_mappings.json",
        language_config_path=f"{config_dir}/language_config.yaml",
        nudge_config_path=f"{config_dir}/nudge_config.yaml"
    )
    report_generator = ReportGenerator(sentiment_analyzer, template_loader)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "supported_languages": config_loader.get_language_codes(),
            "supported_contexts": ["edumentor", "wellness"]
        })
    
    @app.route('/languages', methods=['GET'])
    def get_supported_languages():
        """Get list of supported languages"""
        try:
            languages = config_loader.get_supported_languages()
            return jsonify({
                "languages": languages,
                "fallback_language": config_loader.get_fallback_language(),
                "total_count": len(languages)
            })
        except Exception as e:
            logger.error(f"Error getting supported languages: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @app.route('/generate-report', methods=['POST'])
    def generate_report():
        """
        Generate a multilingual report based on forecast/score data
        """
        try:
            # Validate request
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            # Try to parse JSON
            try:
                payload = request.get_json()
                if payload is None:
                    return jsonify({"error": "Invalid JSON format"}), 400
            except Exception as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return jsonify({"error": "Invalid JSON format"}), 400

            # Validate required fields
            required_fields = ["user_id", "report_type", "context"]
            for field in required_fields:
                if field not in payload:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            # Check for threshold overrides
            override_thresholds = None
            if payload.get("override_thresholds") and payload.get("threshold_overrides"):
                override_thresholds = payload["threshold_overrides"]
                # Validate overrides
                if not config_loader.validate_override_thresholds(override_thresholds):
                    return jsonify({"error": "Invalid threshold overrides"}), 400
                logger.info(f"Using threshold overrides for user {payload['user_id']}")

            # Create report data object
            report_data = ReportData(
                user_id=payload["user_id"],
                report_type=payload["report_type"],
                context=payload["context"],
                language=payload.get("language", "en"),
                data=payload.get("data", {}),
                override_thresholds=override_thresholds
            )

            # Generate report
            report = report_generator.generate_report(report_data)

            # Convert to JSON-serializable format
            response = {
                "report_id": report.report_id,
                "user_id": report.user_id,
                "report_type": report.report_type,
                "language": report.language,
                "title": report.title,
                "content": report.content,
                "sentiment": {
                    "sentiment": report.sentiment["sentiment"].value if hasattr(report.sentiment["sentiment"], 'value') else str(report.sentiment["sentiment"]),
                    "tone": report.sentiment["tone"].value if hasattr(report.sentiment["tone"], 'value') else str(report.sentiment["tone"]),
                    "description": report.sentiment["description"],
                    "urgency": report.sentiment["urgency"],
                    "confidence": report.sentiment["confidence"]
                },
                "nudges": report.nudges,
                "tts_ready": report.tts_ready,
                "metadata": report.metadata,
                "timestamp": report.timestamp
            }

            logger.info(f"Generated report for user {payload['user_id']} in language {report.language}")
            return jsonify(response)

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route('/generate-nudge', methods=['POST'])
    def generate_nudge():
        """
        Generate contextual nudges based on risk levels and user context
        """
        try:
            # Validate request
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            # Try to parse JSON
            try:
                payload = request.get_json()
                if payload is None:
                    return jsonify({"error": "Invalid JSON format"}), 400
            except Exception as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return jsonify({"error": "Invalid JSON format"}), 400

            # Validate required fields
            required_fields = ["user_id", "context", "user_data"]
            for field in required_fields:
                if field not in payload:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            # Create nudge context
            from nudge_engine import NudgeContext, NudgeEngine
            
            nudge_context = NudgeContext(
                user_id=payload["user_id"],
                context=payload["context"],
                user_data=payload["user_data"],
                historical_data=payload.get("historical_data", {}),
                preferences=payload.get("preferences", {}),
                language=payload.get("language", "en")
            )

            # Analyze sentiment based on user data
            score = payload["user_data"].get("score", payload["user_data"].get("average_score", 50))
            sentiment_result = sentiment_analyzer.analyze_score_sentiment(score, nudge_context.language)

            # Generate nudges
            nudge_engine = NudgeEngine(sentiment_analyzer)
            nudges = nudge_engine.generate_nudges(nudge_context, sentiment_result)

            # Convert nudges to JSON-serializable format
            nudges_json = []
            for nudge in nudges:
                nudge_dict = {
                    "nudge_id": nudge.nudge_id,
                    "type": nudge.nudge_type.value,
                    "urgency": nudge.urgency.value,
                    "title": nudge.title,
                    "message": nudge.message,
                    "action_text": nudge.action_text,
                    "action_type": nudge.action_type,
                    "metadata": nudge.metadata,
                    "created_at": nudge.created_at,
                    "expires_at": nudge.expires_at
                }
                nudges_json.append(nudge_dict)

            response = {
                "user_id": payload["user_id"],
                "context": payload["context"],
                "language": nudge_context.language,
                "sentiment": {
                    "sentiment": sentiment_result.sentiment.value,
                    "tone": sentiment_result.tone.value,
                    "description": sentiment_result.description,
                    "urgency": sentiment_result.urgency,
                    "confidence": sentiment_result.confidence
                },
                "nudges": nudges_json,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Generated {len(nudges)} nudges for user {payload['user_id']}")
            return jsonify(response)

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error generating nudges: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route('/generate-lesson', methods=['POST'])
    def generate_multilingual_lesson():
        """
        Generate a multilingual lesson in the new standardized format
        """
        try:
            # Validate request
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            # Try to parse JSON
            try:
                payload = request.get_json()
                if payload is None:
                    return jsonify({"error": "Invalid JSON format"}), 400
            except Exception as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return jsonify({"error": "Invalid JSON format"}), 400

            # Validate required fields
            required_fields = ["subject", "topic"]
            for field in required_fields:
                if field not in payload:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            # Extract parameters
            subject = payload["subject"]
            topic = payload["topic"]
            language = payload.get("language", "en")
            user_id = payload.get("user_id", "guest-user")
            level = payload.get("level", "Seed")

            # Generate lesson content using the report generator
            lesson_report = generate_lesson_report(
                subject=subject,
                topic=topic,
                language=language,
                user_id=user_id,
                level=level,
                report_generator=report_generator,
                sentiment_analyzer=sentiment_analyzer
            )

            logger.info(f"Generated multilingual lesson for {subject}/{topic} in {language}")
            return jsonify(lesson_report)

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error generating lesson: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route('/config/reload', methods=['POST'])
    def reload_config():
        """Reload configuration files"""
        try:
            config_loader.reload_configs()
            template_loader.reload_templates()

            return jsonify({
                "message": "Configuration reloaded successfully",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error reloading config: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 errors"""
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        return jsonify({"error": "Internal server error"}), 500

    return app


def generate_lesson_report(subject: str, topic: str, language: str, user_id: str, level: str,
                          report_generator: ReportGenerator, sentiment_analyzer: SentimentAnalyzer) -> dict:
    """
    Generate a lesson report in the new standardized format with multilingual support

    Args:
        subject: Subject name
        topic: Topic name
        language: Language code (en, hi, bn, etc.)
        user_id: User identifier
        level: Lesson difficulty level
        report_generator: Report generator instance
        sentiment_analyzer: Sentiment analyzer instance

    Returns:
        dict: Lesson in new standardized format
    """
    # Create mock lesson data for educational content
    lesson_data = {
        "user_id": user_id,
        "report_type": "lesson_content",
        "context": "edumentor",
        "language": language,
        "data": {
            "subject": subject,
            "topic": topic,
            "level": level,
            "average_score": 85,  # Mock score for sentiment analysis
            "completed_modules": 5,
            "total_modules": 10,
            "subject_area": topic,
            "weak_areas": f"advanced concepts in {topic}",
            "strong_subjects": f"basic {topic} principles"
        }
    }

    # Create report data object
    report_data = ReportData(
        user_id=user_id,
        report_type="lesson_content",
        context="edumentor",
        language=language,
        data=lesson_data["data"]
    )

    # Generate the base report
    base_report = report_generator.generate_report(report_data)

    # Transform to new lesson format
    lesson_content = transform_report_to_lesson_format(
        base_report=base_report,
        subject=subject,
        topic=topic,
        level=level,
        language=language
    )

    return lesson_content


def transform_report_to_lesson_format(base_report, subject: str, topic: str, level: str, language: str) -> dict:
    """
    Transform a report into the new lesson format

    Args:
        base_report: Generated report from report generator
        subject: Subject name
        topic: Topic name
        level: Lesson level
        language: Language code

    Returns:
        dict: Lesson in new standardized format
    """
    # Generate comprehensive lesson text based on language
    lesson_text = generate_multilingual_lesson_text(subject, topic, language)

    # Generate quiz questions based on language
    quiz_questions = generate_multilingual_quiz(subject, topic, language)

    # Create the new format lesson
    lesson = {
        "title": get_localized_title(subject, topic, language),
        "level": level,
        "text": lesson_text,
        "quiz": quiz_questions,
        "tts": True,
        # Multilingual metadata
        "language": language,
        "subject": subject,
        "topic": topic,
        "sentiment": base_report.sentiment,
        "nudges": base_report.nudges,
        "metadata": {
            **base_report.metadata,
            "lesson_format_version": "2.0",
            "multilingual_support": True,
            "supported_languages": ["en", "hi", "bn", "gu", "mr", "ta", "te", "kn"]
        },
        "timestamp": base_report.timestamp
    }

    return lesson


def generate_multilingual_lesson_text(subject: str, topic: str, language: str) -> str:
    """Generate lesson text in the specified language"""

    # Language-specific lesson templates
    templates = {
        "en": f"""Introduction to {topic}

{topic} is a fundamental concept in {subject} that plays a crucial role in understanding the broader subject matter. This lesson will explore the key principles, applications, and significance of {topic}.

Key Concepts:
- Definition and basic principles of {topic}
- Historical context and development
- Practical applications in real-world scenarios
- Relationship to other concepts in {subject}

Detailed Explanation:
{topic} represents an important area of study within {subject}. Students should focus on understanding both the theoretical foundations and practical applications. The concept builds upon previous knowledge while preparing students for more advanced topics.

Activity:
Engage with {topic} through hands-on exercises, problem-solving activities, and real-world applications. Practice identifying examples of {topic} in everyday situations.

Reflection Question:
How does understanding {topic} enhance your overall comprehension of {subject}? What connections can you make between {topic} and other areas of study?""",

        "hi": f"""{topic} का परिचय

{topic} {subject} में एक मौलिक अवधारणा है जो व्यापक विषय वस्तु को समझने में महत्वपूर्ण भूमिका निभाती है। यह पाठ {topic} के मुख्य सिद्धांतों, अनुप्रयोगों और महत्व की खोज करेगा।

मुख्य अवधारणाएं:
- {topic} की परिभाषा और बुनियादी सिद्धांत
- ऐतिहासिक संदर्भ और विकास
- वास्तविक दुनिया के परिदृश्यों में व्यावहारिक अनुप्रयोग
- {subject} में अन्य अवधारणाओं के साथ संबंध

विस्तृत व्याख्या:
{topic} {subject} के भीतर अध्ययन का एक महत्वपूर्ण क्षेत्र है। छात्रों को सैद्धांतिक आधार और व्यावहारिक अनुप्रयोगों दोनों को समझने पर ध्यान देना चाहिए। यह अवधारणा पिछले ज्ञान पर आधारित है और छात्रों को अधिक उन्नत विषयों के लिए तैयार करती है।

गतिविधि:
व्यावहारिक अभ्यास, समस्या-समाधान गतिविधियों और वास्तविक दुनिया के अनुप्रयोगों के माध्यम से {topic} के साथ जुड़ें। रोजमर्रा की स्थितियों में {topic} के उदाहरणों की पहचान करने का अभ्यास करें।

चिंतन प्रश्न:
{topic} को समझना {subject} की आपकी समग्र समझ को कैसे बढ़ाता है? आप {topic} और अध्ययन के अन्य क्षेत्रों के बीच क्या संबंध बना सकते हैं?""",

        "bn": f"""{topic} এর পরিচয়

{topic} হল {subject} এর একটি মৌলিক ধারণা যা বিস্তৃত বিষয়বস্তু বোঝার ক্ষেত্রে গুরুত্বপূর্ণ ভূমিকা পালন করে। এই পাঠটি {topic} এর মূল নীতি, প্রয়োগ এবং তাৎপর্য অন্বেষণ করবে।

মূল ধারণাসমূহ:
- {topic} এর সংজ্ঞা এবং মৌলিক নীতি
- ঐতিহাসিক প্রেক্ষাপট এবং উন্নয়ন
- বাস্তব জগতের পরিস্থিতিতে ব্যবহারিক প্রয়োগ
- {subject} এর অন্যান্য ধারণার সাথে সম্পর্ক

বিস্তারিত ব্যাখ্যা:
{topic} {subject} এর মধ্যে অধ্যয়নের একটি গুরুত্বপূর্ণ ক্ষেত্র। শিক্ষার্থীদের তাত্ত্বিক ভিত্তি এবং ব্যবহারিক প্রয়োগ উভয়ই বোঝার দিকে মনোনিবেশ করা উচিত। ধারণাটি পূর্ববর্তী জ্ঞানের উপর ভিত্তি করে এবং শিক্ষার্থীদের আরও উন্নত বিষয়ের জন্য প্রস্তুত করে।

কার্যকলাপ:
হাতে-কলমে অনুশীলন, সমস্যা সমাধানের কার্যকলাপ এবং বাস্তব জগতের প্রয়োগের মাধ্যমে {topic} এর সাথে জড়িত হন। দৈনন্দিন পরিস্থিতিতে {topic} এর উদাহরণ চিহ্নিত করার অনুশীলন করুন।

চিন্তাভাবনার প্রশ্ন:
{topic} বোঝা কীভাবে {subject} সম্পর্কে আপনার সামগ্রিক বোধগম্যতা বাড়ায়? আপনি {topic} এবং অধ্যয়নের অন্যান্য ক্ষেত্রের মধ্যে কী সংযোগ তৈরি করতে পারেন?"""
    }

    return templates.get(language, templates["en"])


def generate_multilingual_quiz(subject: str, topic: str, language: str) -> list:
    """Generate quiz questions in the specified language"""

    quiz_templates = {
        "en": [
            {
                "question": f"What is the main focus of this lesson about {topic}?",
                "options": [
                    f"Understanding the fundamental concepts of {topic}",
                    f"Memorizing facts about {subject}",
                    f"Learning unrelated information",
                    f"Avoiding the topic entirely"
                ],
                "correct": 0,
                "explanation": f"This lesson focuses on understanding the fundamental concepts and applications of {topic} in {subject}."
            },
            {
                "question": f"How does {topic} relate to {subject}?",
                "options": [
                    f"{topic} is a fundamental concept in {subject}",
                    f"{topic} is unrelated to {subject}",
                    f"{topic} contradicts {subject} principles",
                    f"{topic} is more important than {subject}"
                ],
                "correct": 0,
                "explanation": f"{topic} is indeed a fundamental concept that plays a crucial role in understanding {subject}."
            }
        ],
        "hi": [
            {
                "question": f"{topic} के बारे में इस पाठ का मुख्य फोकस क्या है?",
                "options": [
                    f"{topic} की मौलिक अवधारणाओं को समझना",
                    f"{subject} के बारे में तथ्यों को याद करना",
                    f"असंबंधित जानकारी सीखना",
                    f"विषय से पूरी तरह बचना"
                ],
                "correct": 0,
                "explanation": f"यह पाठ {subject} में {topic} की मौलिक अवधारणाओं और अनुप्रयोगों को समझने पर केंद्रित है।"
            },
            {
                "question": f"{topic} का {subject} से क्या संबंध है?",
                "options": [
                    f"{topic} {subject} में एक मौलिक अवधारणा है",
                    f"{topic} का {subject} से कोई संबंध नहीं है",
                    f"{topic} {subject} के सिद्धांतों का विरोध करता है",
                    f"{topic} {subject} से अधिक महत्वपूर्ण है"
                ],
                "correct": 0,
                "explanation": f"{topic} वास्तव में एक मौलिक अवधारणा है जो {subject} को समझने में महत्वपूर्ण भूमिका निभाती है।"
            }
        ],
        "bn": [
            {
                "question": f"{topic} সম্পর্কে এই পাঠের মূল ফোকাস কী?",
                "options": [
                    f"{topic} এর মৌলিক ধারণাগুলি বোঝা",
                    f"{subject} সম্পর্কে তথ্য মুখস্থ করা",
                    f"অসম্পর্কিত তথ্য শেখা",
                    f"বিষয়টি সম্পূর্ণভাবে এড়িয়ে যাওয়া"
                ],
                "correct": 0,
                "explanation": f"এই পাঠটি {subject} এ {topic} এর মৌলিক ধারণা এবং প্রয়োগ বোঝার উপর দৃষ্টি নিবদ্ধ করে।"
            },
            {
                "question": f"{topic} এর সাথে {subject} এর কী সম্পর্ক?",
                "options": [
                    f"{topic} হল {subject} এর একটি মৌলিক ধারণা",
                    f"{topic} এর সাথে {subject} এর কোনো সম্পর্ক নেই",
                    f"{topic} {subject} এর নীতির বিরোধিতা করে",
                    f"{topic} {subject} এর চেয়ে বেশি গুরুত্বপূর্ণ"
                ],
                "correct": 0,
                "explanation": f"{topic} প্রকৃতপক্ষে একটি মৌলিক ধারণা যা {subject} বোঝার ক্ষেত্রে গুরুত্বপূর্ণ ভূমিকা পালন করে।"
            }
        ]
    }

    return quiz_templates.get(language, quiz_templates["en"])


def get_localized_title(subject: str, topic: str, language: str) -> str:
    """Get localized title for the lesson"""

    title_templates = {
        "en": f"Introduction to {topic}",
        "hi": f"{topic} का परिचय",
        "bn": f"{topic} এর পরিচয়",
        "gu": f"{topic} નો પરિચય",
        "mr": f"{topic} चा परिचय",
        "ta": f"{topic} அறிமுகம்",
        "te": f"{topic} పరిచయం",
        "kn": f"{topic} ಪರಿಚಯ"
    }

    return title_templates.get(language, title_templates["en"])


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
