/**
 * Simple test to demonstrate the verbose_call functionality
 * This simulates the generateCodeSnippet function with verbose_call metadata
 */

// Mock function with verbose_call metadata (similar to matched_filter_coherent_shm)
const mockFunction = {
    name: 'matched_filter_coherent_shm',
    displayName: 'Coherent Matched Filter',
    module: 'shmtools.active_sensing.matched_filter',
    description: 'Computes coherent matched filter for waveform analysis.',
    parameters: [
        {
            name: 'waveform',
            type: 'array_like',
            optional: false,
            default: null,
            description: 'Input waveform data'
        },
        {
            name: 'matched_waveform', 
            type: 'array_like',
            optional: false,
            default: null,
            description: 'Template waveform for matching'
        }
    ],
    guiMetadata: {
        verbose_call: 'Filter Result = Coherent Matched Filter (Waveform, Matched Waveform)',
        category: 'Feature Extraction - Active Sensing',
        matlab_equivalent: 'coherentMatchedFilter_shm'
    },
    returns: [
        {
            name: 'filter_result',
            type: 'ndarray',
            description: 'Coherent matched filter output'
        }
    ]
};

// Simulate the generateCodeSnippet logic
function generateMockCodeSnippet(func) {
    // Function header
    let code = `# ${func.description}\n`;
    
    // Add verbose_call comment if available (OUR NEW FEATURE)
    if (func.guiMetadata && func.guiMetadata.verbose_call) {
        code += `# ${func.guiMetadata.verbose_call}\n`;
    }
    
    // Generate parameters
    let paramStrings = [];
    func.parameters.forEach((param, index) => {
        let paramStr = `    ${param.name}=None`;  // Simplified for demo
        let comment = param.description || param.name;
        
        if (index < func.parameters.length - 1) {
            paramStr += `,  # ${comment}`;
        } else {
            paramStr += `  # ${comment}`;
        }
        
        paramStrings.push(paramStr);
    });
    
    // Generate function call
    const outputVar = func.returns && func.returns.length > 0 ? func.returns[0].name : 'result';
    
    if (paramStrings.length > 0) {
        code += `${outputVar} = ${func.module}.${func.name}(\n${paramStrings.join('\n')}\n)`;
    } else {
        code += `${outputVar} = ${func.module}.${func.name}()`;
    }
    
    return code;
}

// Test the implementation
console.log("=== Generated Code Snippet ===");
const snippet = generateMockCodeSnippet(mockFunction);
console.log(snippet);

console.log("\n=== Expected Output ===");
console.log("The generated code should include the verbose_call comment:");
console.log("'# Filter Result = Coherent Matched Filter (Waveform, Matched Waveform)'");
console.log("positioned just before the function call.");