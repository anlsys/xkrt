/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "XKRT", "index.html", [
    [ "XKaapi Runtime (XKRT) - A parallel runtime system for macro-dataflow on multi-devices architectures.", "index.html#autotoc_md0", null ],
    [ "Related Projects", "index.html#autotoc_md1", null ],
    [ "Software Versions compatibility", "index.html#autotoc_md2", null ],
    [ "Installation", "index.html#autotoc_md3", [
      [ "Requirements of XKRT", "index.html#autotoc_md4", null ],
      [ "Optional", "index.html#autotoc_md5", null ],
      [ "Build (One line assistant)", "index.html#autotoc_md6", null ],
      [ "Build example (Manual)", "index.html#autotoc_md7", null ]
    ] ],
    [ "Available environment variable", "index.html#autotoc_md8", null ],
    [ "Directions for improvements / known issues", "index.html#autotoc_md9", null ],
    [ "Examples", "examples.html", [
      [ "AXPY – Vector Addition with Data Accesses", "examples.html#autotoc_md11", [
        [ "Key concepts demonstrated", "examples.html#autotoc_md12", null ]
      ] ],
      [ "AXPBY – Moldable Tasks with Split Conditions", "examples.html#autotoc_md14", [
        [ "Key concepts demonstrated", "examples.html#autotoc_md15", null ]
      ] ],
      [ "GPU Kernel Launch", "examples.html#autotoc_md17", [
        [ "Key concepts demonstrated", "examples.html#autotoc_md18", null ]
      ] ],
      [ "Fibonacci – Recursive Tasking with Teams", "examples.html#autotoc_md20", [
        [ "Key concepts demonstrated", "examples.html#autotoc_md21", null ]
      ] ],
      [ "File-to-GPU I/O Pipeline", "examples.html#autotoc_md23", [
        [ "Key concepts demonstrated", "examples.html#autotoc_md24", null ]
      ] ],
      [ "2D Heat Diffusion Stencil", "examples.html#autotoc_md26", null ]
    ] ],
    [ "Architecture & Concepts", "architecture.html", [
      [ "Overview", "architecture.html#autotoc_md28", null ],
      [ "Software Stack", "architecture.html#autotoc_md30", null ],
      [ "Tasks", "architecture.html#autotoc_md32", [
        [ "Task Flags", "architecture.html#autotoc_md33", null ],
        [ "Task States", "architecture.html#autotoc_md34", null ],
        [ "Task Memory Layout", "architecture.html#autotoc_md35", null ],
        [ "Moldable Tasks", "architecture.html#autotoc_md36", null ],
        [ "Task Formats", "architecture.html#autotoc_md37", null ],
        [ "Task Dependency Graphs (TDG)", "architecture.html#autotoc_md38", null ]
      ] ],
      [ "Data Accesses", "architecture.html#autotoc_md40", [
        [ "Access Modes", "architecture.html#autotoc_md41", null ],
        [ "Mapping to OpenMP 5.x/6.0 Dependency Types", "architecture.html#autotoc_md42", null ],
        [ "Access Concurrency", "architecture.html#autotoc_md43", null ],
        [ "Access Scope", "architecture.html#autotoc_md44", null ],
        [ "Access Types", "architecture.html#autotoc_md45", null ],
        [ "Dependency Resolution", "architecture.html#autotoc_md46", null ]
      ] ],
      [ "Memory Coherence", "architecture.html#autotoc_md48", [
        [ "Memory Registration (Pinning)", "architecture.html#autotoc_md49", null ],
        [ "Data Distribution", "architecture.html#autotoc_md50", null ]
      ] ],
      [ "Drivers", "architecture.html#autotoc_md52", null ],
      [ "Thread Teams", "architecture.html#autotoc_md54", null ],
      [ "Constants", "architecture.html#autotoc_md56", null ],
      [ "Configuration", "architecture.html#autotoc_md58", null ]
    ] ],
    [ "Building & Configuration", "building.html", [
      [ "</blockquote>", "building.html#autotoc_md59", null ],
      [ "Requirements", "building.html#autotoc_md60", null ],
      [ "Optional Dependencies", "building.html#autotoc_md61", [
        [ "GPU Backends", "building.html#autotoc_md62", null ],
        [ "BLAS Libraries", "building.html#autotoc_md63", null ],
        [ "Management / Monitoring", "building.html#autotoc_md64", null ],
        [ "Other", "building.html#autotoc_md65", null ]
      ] ],
      [ "Build Examples", "building.html#autotoc_md67", [
        [ "Host-Only (Development / No GPU)", "building.html#autotoc_md68", null ],
        [ "With CUDA Support", "building.html#autotoc_md69", null ],
        [ "With CUDA (Optimized Release)", "building.html#autotoc_md70", null ],
        [ "With Multiple Backends", "building.html#autotoc_md71", null ]
      ] ],
      [ "CMake Options Reference", "building.html#autotoc_md73", null ],
      [ "Build Outputs", "building.html#autotoc_md75", null ],
      [ "Installation", "building.html#autotoc_md77", [
        [ "Using XKRT from CMake", "building.html#autotoc_md78", null ]
      ] ],
      [ "Running Tests", "building.html#autotoc_md80", [
        [ "Test layout & labels", "building.html#autotoc_md81", null ]
      ] ],
      [ "Environment Variables", "building.html#autotoc_md83", null ]
    ] ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ],
        [ "Enumerator", "namespacemembers_eval.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", null ],
        [ "Variables", "functions_vars.html", null ],
        [ "Typedefs", "functions_type.html", null ],
        [ "Enumerations", "functions_enum.html", null ],
        [ "Enumerator", "functions_eval.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", "globals_dup" ],
        [ "Functions", "globals_func.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Enumerations", "globals_enum.html", null ],
        [ "Enumerator", "globals_eval.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"annotated.html",
"namespacexkrt.html#a3cbdfc451bd31c7a842765dc8d63cd9a",
"structxkrt_1_1runtime__t.html#a072e6aa83b25d0b1f99eb7ece99d75ac",
"structxkrt_1_1runtime__t.html#ae0afdaa800b6a8bb6e73b602a585857b"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';