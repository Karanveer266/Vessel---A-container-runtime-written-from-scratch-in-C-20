import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// gRPC client layer. Deliberately a plain Kotlin/JVM module rather than an
// Android library: nothing here needs the Android SDK (grpc-okhttp runs on any
// JVM), so codegen is simpler, and the in-process gRPC tests run as ordinary
// JVM tests without Robolectric. Android-only niceties (network-change aware
// channels) are layered on in :app via the ChannelFactory seam.
plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.protobuf)
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

sourceSets {
    main {
        proto {
            // Single source of truth shared with vesseld/vesselctl: compile the
            // repo's proto in place (android/../proto), never a copy.
            srcDir(rootDir.resolve("../proto"))
        }
    }
}

protobuf {
    protoc { artifact = libs.protobuf.protoc.get().toString() }
    plugins {
        create("grpc") { artifact = libs.grpc.protoc.gen.java.get().toString() }
        create("grpckt") { artifact = "${libs.grpc.protoc.gen.kotlin.get()}:jdk8@jar" }
    }
    generateProtoTasks {
        all().configureEach {
            builtins {
                named("java") { option("lite") }
                create("kotlin") { option("lite") }
            }
            plugins {
                create("grpc") { option("lite") }
                create("grpckt") { option("lite") }
            }
        }
    }
}

dependencies {
    api(project(":core:model"))

    api(libs.protobuf.javalite)
    api(libs.protobuf.kotlin.lite)
    api(libs.grpc.stub)
    api(libs.grpc.protobuf.lite)
    api(libs.grpc.kotlin.stub)
    implementation(libs.grpc.okhttp)

    testImplementation(libs.junit)
    testImplementation(libs.kotlin.test)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.turbine)
    testImplementation(libs.grpc.inprocess)
    testImplementation(libs.grpc.testing)
}
