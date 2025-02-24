#!/bin/bash

# Find all markdown files
find . -name "*.md" | while read -r file; do
    echo "Checking $file..."
    
    # Count opening and closing code fences
    opens=$(grep -c "^\`\`\`" "$file")
    closes=$(grep -c "^\`\`\`$" "$file")
    
    if [ "$opens" != "$closes" ]; then
        echo "WARNING: Unbalanced code blocks in $file"
        echo "Opening blocks: $opens"
        echo "Closing blocks: $closes"
    fi
done 