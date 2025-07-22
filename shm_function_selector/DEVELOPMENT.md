# JupyterLab Extension Development Guide

## CRITICAL: Proper Build Process

The JupyterLab extension has a **mandatory 3-step build process**. Skipping any step will result in changes not appearing.

### Step-by-Step Build Process

```bash
# ALWAYS start from the extension directory
cd shm_function_selector/

# Step 1: Compile TypeScript to JavaScript (REQUIRED FIRST)
npm run build:lib

# Step 2: Build the JupyterLab extension (uses compiled JS from Step 1)
npm run build:labextension:dev

# Step 3: Integrate extension into JupyterLab (from parent directory)
cd ..
source venv/bin/activate
jupyter lab build
```

### Common Mistakes That Waste Time

❌ **NEVER DO THESE**:
- Skip `npm run build:lib` - TypeScript changes won't compile
- Run only `jupyter lab build` - won't pick up extension changes
- Work from wrong directory - commands will fail
- Forget to activate virtual environment for Step 3

✅ **ALWAYS DO THESE**:
- Run all 3 steps in order after TypeScript changes
- Check file hash changes to confirm new builds
- Clear cache if changes don't appear
- Refresh browser after `jupyter lab build`

### Debugging Build Issues

#### 1. Verify TypeScript Compilation
```bash
cd shm_function_selector/
npm run build:lib
# Check if your changes appear:
grep "your_debug_text" lib/index.js
```

#### 2. Verify Extension Build
```bash
npm run build:labextension:dev
# Check if compiled JS made it to extension:
grep "your_debug_text" shm_function_selector/labextension/static/lib_index_js.*.js
```

#### 3. Force Complete Rebuild
If builds appear stuck or changes don't show:
```bash
# Clear all cached builds
rm -rf shm_function_selector/labextension/static/*.js

# Rebuild everything from scratch
npm run build:lib
npm run build:labextension:dev
cd .. && source venv/bin/activate && jupyter lab build
```

#### 4. Verify JupyterLab Integration
```bash
# Check that JupyterLab recognizes the extension
jupyter labextension list | grep shm-function-selector
```

### Development Workflow

#### Making Code Changes
1. Edit TypeScript files in `src/`
2. Run the 3-step build process
3. Refresh JupyterLab browser tab
4. Test changes
5. Check browser console for debugging output

#### Adding Debug Output
Always add debug console.log statements when debugging:
```typescript
console.log('🔍 Your debug message:', variable);
```

Then verify the debug output appears in the built file:
```bash
grep "Your debug message" shm_function_selector/labextension/static/lib_index_js.*.js
```

### File Structure
```
shm_function_selector/
├── src/index.ts                    # Main TypeScript source
├── lib/index.js                    # Compiled JavaScript (auto-generated)
├── shm_function_selector/
│   └── labextension/
│       └── static/
│           └── lib_index_js.*.js   # Webpack bundle (auto-generated)
└── package.json                    # Build scripts
```

### Build Scripts Explained
- `npm run build:lib` → Runs `tsc --sourceMap` (TypeScript compiler)
- `npm run build:labextension:dev` → Runs `jupyter labextension build --development True .`
- Final step: `jupyter lab build` integrates extension into JupyterLab

### Time-Saving Tips
1. **Use file hashes**: Built files have hashes like `lib_index_js.abc123.js` - hash changes confirm new build
2. **Check timestamps**: Use `ls -la shm_function_selector/labextension/static/` to verify recent builds
3. **Use grep**: Always verify your changes made it through the build pipeline
4. **Browser dev tools**: Check Network tab to confirm JupyterLab loaded new extension file

Remember: **Every build failure wastes time and delays progress. Follow this process exactly.**