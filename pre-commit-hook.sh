#!/bin/bash
 
set -e
 
echo "🔧 Installing pre-commit..."
sudo apt update && sudo apt install -y pre-commit
 
echo "📁 Creating .git/hooks/pre-commit custom AWS key block hook..."
 
# Make sure we're inside a git repo
if [ ! -d ".git" ]; then
  echo "❌ This is not a Git repository. Please run this inside your repo root."
  exit 1
fi
 
# Create .git/hooks/pre-commit
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
 
# Regex patterns for AWS keys
ACCESS_KEY_PATTERN='AKIA[0-9A-Z]{12,20}'
SECRET_KEY_PATTERN='(?i)aws(.{0,20})?(secret|private)?(.{0,20})?["'"'"'][0-9a-zA-Z/+]{40}["'"'"']'
 
# Get staged files
FILES=$(git diff --cached --name-only --diff-filter=ACM)
 
for FILE in $FILES; do
  if [ -f "$FILE" ]; then
    # Only check text files
    if file "$FILE" | grep -q text; then
      if grep -E -q "$ACCESS_KEY_PATTERN" "$FILE"; then
        echo "❌ Potential AWS Access Key found in $FILE"
        exit 1
      fi
 
      if grep -P -q "$SECRET_KEY_PATTERN" "$FILE"; then
        echo "❌ Potential AWS Secret Key found in $FILE"
        exit 1
      fi
    fi
  fi
done
 
exit 0
EOF
 
chmod +x .git/hooks/pre-commit
 
echo "✅ Pre-commit hook installed to block AWS credentials."