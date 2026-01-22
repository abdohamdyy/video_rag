"""
Script to create ElevenLabs Knowledge Base and get IDs.
Run this script after setting up your .env file with ELEVENLABS_API_KEY.
"""

from app.elevenlabs_knowledge_base import create_knowledge_base
from app.settings import get_settings

def main():
    settings = get_settings()
    
    if not settings.elevenlabs_api_key:
        print("❌ Error: ELEVENLABS_API_KEY not found in .env file")
        print("Please add your ElevenLabs API key to .env file")
        return
    
    print("🚀 Creating ElevenLabs Knowledge Base...")
    print(f"API Key: {settings.elevenlabs_api_key[:10]}...")
    
    try:
        # Create Knowledge Base
        result = create_knowledge_base(
            api_key=settings.elevenlabs_api_key,
            name="Technical Support Knowledge Base"
        )
        
        print("\n✅ Knowledge Base created successfully!")
        print(f"📋 Knowledge Base ID: {result['id']}")
        print(f"📝 Name: {result['name']}")
        
        print("\n" + "="*60)
        print("📝 Next Steps:")
        print("="*60)
        print(f"\n1. Add this to your .env file:")
        print(f"   ELEVENLABS_KNOWLEDGE_BASE_ID={result['id']}")
        print("\n2. Go to ElevenLabs Dashboard:")
        print("   https://elevenlabs.io/app/agents")
        print("\n3. Create a new Agent:")
        print("   - Name: Technical Support Agent")
        print("   - Voice: Choose multilingual voice")
        print(f"   - Knowledge Base: Select '{result['name']}'")
        print("   - Enable RAG: ✅ Yes")
        print("\n4. Copy the Agent ID and add to .env:")
        print("   ELEVENLABS_AGENT_ID=your_agent_id_here")
        print("\n5. Restart your FastAPI server")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error creating Knowledge Base: {e}")
        print("\n" + "="*60)
        print("📝 Alternative: Create Knowledge Base Manually")
        print("="*60)
        print("\nSince API creation may not be available, please:")
        print("\n1. Go to ElevenLabs Dashboard:")
        print("   https://elevenlabs.io/app/knowledge-base")
        print("\n2. Click 'Create Knowledge Base' or 'New Knowledge Base'")
        print("3. Name it: 'Technical Support Knowledge Base'")
        print("4. Copy the Knowledge Base ID")
        print("5. Add it to your .env file:")
        print("   ELEVENLABS_KNOWLEDGE_BASE_ID=your_kb_id_here")
        print("\n6. Then create an Agent:")
        print("   https://elevenlabs.io/app/agents")
        print("   - Link it to the Knowledge Base")
        print("   - Enable RAG")
        print("   - Copy Agent ID and add to .env:")
        print("   ELEVENLABS_AGENT_ID=your_agent_id_here")
        print("\nTroubleshooting:")
        print("1. Check your ELEVENLABS_API_KEY in .env")
        print("2. Make sure you have an active ElevenLabs account")
        print("3. Check your internet connection")
        print("4. Ensure you have access to Agents Platform features")

if __name__ == "__main__":
    main()
