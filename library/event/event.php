<?php

/**
 * Description of event
 *
 * @author durso
 */

namespace library\event;
use library\dom\dom;
use app\request;

class event {
    private $type;
    private $source;
    private static $eventList = array("click","submit","location");
    
    public function __construct($type,$source){
        assert(request::isAjax());
        if(self::isEvent($type)){
            $this->type = $type;
        } else {
            throw new \Exception("Event not supported");
        }
        dom::load();
        $this->source = dom::getElementById($source);
        
    }
    
    public function getSource(){
        return $this->source;
    }
    public function getType(){
        return $this->type;
    }
    public function trigger(){
        $listener = dom::getEventListener($this->source->getId(),$this->type);
        $listener->fire($this);
    }
    
    public static function isEvent($event){
        return in_array($event, self::$eventList);
    }
}
