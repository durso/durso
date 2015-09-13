<?php

/**
 * Description of listener
 *
 * @author durso
 */

namespace library\event;
use library\dom\elements\element;
use library\event\event;
use library\dom\javascript;



class listener {
    protected $components = array();
    
    public function __construct(element $component,$event,$callback,$args){
        $this->register($component,$event,$callback,$args);
    }
    
    public function register(element $component,$event,$callback,$args){
        assert(event::isEvent($event));
        $this->components[$component->getId()][$event]["callback"] = $callback;
        $this->components[$component->getId()][$event]["args"] = $args;
    }
    
    public function fire(event $event){
        $id = $event->getSource()->getId();
        $evt = $event->getType();
        javascript::init($event->getSource());
        if(isset($this->components[$id][$evt])){
            if(empty($this->components[$id][$evt]["args"])){
                call_user_func($this->components[$id][$evt]['callback']);
            } else {
                call_user_func_array($this->components[$id][$evt]['callback'], $this->components[$id][$evt]['args']);
            }
        }
        
    }
}
