# protobuf-javalite reflects over generated message fields.
-keep class * extends com.google.protobuf.GeneratedMessageLite { <fields>; }

# grpc-okhttp / grpc-core optional dependencies that are not on the classpath.
-dontwarn com.google.errorprone.annotations.**
-dontwarn javax.naming.**
-dontwarn org.codehaus.mojo.animal_sniffer.**
-dontwarn com.squareup.okhttp.**
-dontwarn okio.**
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**

# grpc service loader providers.
-keep class io.grpc.okhttp.OkHttpChannelProvider { *; }
-keep class io.grpc.internal.PickFirstLoadBalancerProvider { *; }
-keep class io.grpc.internal.DnsNameResolverProvider { *; }
