import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// Pure Kotlin/JVM: domain models + platform-free primitives (ring buffers,
// formatters, flow operators). No Android, no protobuf, no gRPC — this is the
// vocabulary shared by :core:api and :app, and its tests run in milliseconds.
plugins {
    alias(libs.plugins.kotlin.jvm)
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
        freeCompilerArgs.add("-Xjsr305=strict")
    }
}

dependencies {
    api(libs.kotlinx.coroutines.core)

    testImplementation(libs.junit)
    testImplementation(libs.kotlin.test)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.turbine)
}
