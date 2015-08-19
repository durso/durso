<?php
/**
 * Description of javascript
 *
 * @author durso
 */
namespace library\dom;
use library\dom\elements\element;
use library\dom\object;
use app\response;


class javascript{
    private static $dom;
    private static $response;
    private static $context = null;
    
    public static function init(element $component){
        self::$dom = dom::getDocument();
        self::$response = array();
        self::$context = $component;
    }
    public static function isActive(){
        return !is_null(self::$context);
    }
    public static function update(object $component,$method,$value = false,$key = false){
        if(self::$context === $component){
            $context = 'this';
        } else {
            $context = $component->getUid();
        }
        if(!$value && !$key){
            self::addMethod($context,$method);
        } elseif($value && $key === false){
            self::addValue($value, $context, $method);
        } else {
            self::addKeyValue($key, $value, $context, $method);
        }
    }
    private static function addMethod($context,$method){
        self::$response[] = array("context" => $context,"method" => $method);
    }
    private static function addValue($value,$context,$method){
        self::$response[] = array("context" => $context,"method" => $method, "value" => sprintf($value));
    }
    private static function addKeyValue($key,$value,$context,$method){
        self::$response[] = array("context" => $context,"method" => $method,"key" =>$key, "value" => $value);
    }
    public function clear(){
        self::$response = array();
    }
    public static function getResponse(){
        dom::update();
        response::json();
        return json_encode(self::$response);
    }

}
